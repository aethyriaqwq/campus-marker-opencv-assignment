#include "detector.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace {

cv::Mat column(const cv::Vec3d& v) {
  return (cv::Mat_<double>(3, 1) << v[0], v[1], v[2]);
}

cv::Point2d rotate90(cv::Point2d p, double center, int turns) {
  double x = p.x - center;
  double y = p.y - center;
  for (int i = 0; i < turns; ++i) {
    const double nx = -y;
    const double ny = x;
    x = nx;
    y = ny;
  }
  return {x + center, y + center};
}

float gray_at(const cv::Mat& gray, double x, double y) {
  if (x < 1.0 || y < 1.0 || x >= gray.cols - 2.0 || y >= gray.rows - 2.0) {
    return -1.f;
  }
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const float fx = static_cast<float>(x - x0);
  const float fy = static_cast<float>(y - y0);
  const float v00 = gray.at<uchar>(y0, x0);
  const float v10 = gray.at<uchar>(y0, x0 + 1);
  const float v01 = gray.at<uchar>(y0 + 1, x0);
  const float v11 = gray.at<uchar>(y0 + 1, x0 + 1);
  return (1.f - fx) * (1.f - fy) * v00 + fx * (1.f - fy) * v10 + (1.f - fx) * fy * v01 + fx * fy * v11;
}

double median_of(std::vector<double> values) {
  if (values.empty()) {
    return 0;
  }
  const std::size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
  return values[mid];
}

double shoelace(const std::array<cv::Point2d, 4>& p) {
  double sum = 0;
  for (int i = 0; i < 4; ++i) {
    const cv::Point2d& a = p[static_cast<std::size_t>(i)];
    const cv::Point2d& b = p[static_cast<std::size_t>((i + 1) % 4)];
    sum += a.x * b.y - b.x * a.y;
  }
  return 0.5 * sum;
}

}  // namespace

bool MarkerDetector::load(const std::string& model_path, const std::string& params_path, const cv::Mat& camera,
                          const cv::Mat& dist, std::string& error) {
  ready_ = false;
  clear_track();
  if (!load_model(model_path, model_, error) || !load_params(params_path, params_, error)) {
    return false;
  }
  if (model_.notch_gap - 2.0 * params_.run_margin < params_.min_samples) {
    error = "run_margin leaves no interior on the notch";
    return false;
  }
  if (camera.empty() || camera.rows != 3 || camera.cols != 3) {
    error = "camera matrix missing";
    return false;
  }
  camera.convertTo(camera_, CV_64F);
  if (dist.empty()) {
    dist_ = cv::Mat::zeros(1, 5, CV_64F);
  } else {
    dist.convertTo(dist_, CV_64F);
  }
  edges_.clear();
  // Only the outer square. Hole-facing and notch-internal edges are real
  // borders of the lamps, but a blur bias there pulls the plate yaw while
  // the reported corners are the outer ones.
  const double outer = model_.outer;
  auto on_plate = [&](const Boundary& segment) {
    const bool horizontal = std::abs(segment.a.y - segment.b.y) < 1e-6;
    const double coordinate = horizontal ? segment.a.y : segment.a.x;
    return std::abs(coordinate) < 1e-6 || std::abs(coordinate - outer) < 1e-6;
  };
  for (const Boundary& segment : model_.boundary) {
    if (!on_plate(segment)) {
      continue;
    }
    const double length = cv::norm(segment.b - segment.a);
    const int count = std::max(1, static_cast<int>(std::lround(length / params_.sample_step)));
    int side = 0;
    if (segment.outward.x > 0.5) {
      side = 1;
    } else if (segment.outward.y > 0.5) {
      side = 2;
    } else if (segment.outward.x < -0.5) {
      side = 3;
    }
    for (int i = 0; i < count; ++i) {
      const double t = (static_cast<double>(i) + 0.5) / static_cast<double>(count);
      edges_.push_back({segment.a + (segment.b - segment.a) * t, segment.outward, side});
    }
  }
  interior_.clear();
  for (int i = 0; i < static_cast<int>(model_.samples.size()); ++i) {
    const Sample& sample = model_.samples[static_cast<std::size_t>(i)];
    const Run& run = model_.runs[static_cast<std::size_t>(sample.run)];
    if (sample.coord >= run.a + params_.run_margin && sample.coord <= run.b - params_.run_margin) {
      interior_.push_back(i);
    }
  }
  if (edges_.empty() || interior_.empty()) {
    error = "marker has no readable samples";
    return false;
  }
  ready_ = true;
  return true;
}

std::vector<cv::Point2d> MarkerDetector::project(const Pose& pose, const std::vector<cv::Point3d>& object) const {
  std::vector<cv::Point2d> image;
  if (!object.empty()) {
    cv::projectPoints(object, column(pose.r), column(pose.t), camera_, dist_, image);
  }
  return image;
}

std::array<cv::Point2d, 4> MarkerDetector::plate(const Pose& pose) const {
  std::vector<cv::Point3d> object;
  object.reserve(4);
  for (const cv::Point2d& corner : model_.corners) {
    object.push_back({corner.x, corner.y, 0});
  }
  const std::vector<cv::Point2d> image = project(pose, object);
  std::array<cv::Point2d, 4> corners{};
  if (image.size() == 4) {
    for (int i = 0; i < 4; ++i) {
      corners[static_cast<std::size_t>(i)] = image[static_cast<std::size_t>(i)];
    }
  }
  return corners;
}

bool MarkerDetector::facing(const Pose& pose) const {
  cv::Mat rotation;
  cv::Rodrigues(column(pose.r), rotation);
  for (const cv::Point2d& corner : model_.corners) {
    const cv::Mat local = (cv::Mat_<double>(3, 1) << corner.x, corner.y, 0);
    const cv::Mat world = rotation * local + column(pose.t);
    if (world.at<double>(2) <= 1.0) {
      return false;
    }
  }
  const std::array<cv::Point2d, 4> corners = plate(pose);
  for (const cv::Point2d& corner : corners) {
    if (!std::isfinite(corner.x) || !std::isfinite(corner.y)) {
      return false;
    }
  }
  return shoelace(corners) > 0;
}

MarkerDetector::Orientation MarkerDetector::orient(const Pose& pose, const cv::Mat& gray, bool notch_block) const {
  Orientation result;
  if (interior_.empty()) {
    return result;
  }
  const double center = model_.outer * 0.5;
  std::vector<double> bright_c;
  std::vector<double> dark_c;
  for (int turn = 0; turn < 4; ++turn) {
    std::vector<cv::Point3d> at_obj;
    std::vector<cv::Point3d> in_obj;
    at_obj.reserve(interior_.size());
    in_obj.reserve(interior_.size());
    for (const int index : interior_) {
      const Sample& sample = model_.samples[static_cast<std::size_t>(index)];
      const cv::Point2d at = rotate90(sample.at, center, turn);
      const cv::Point2d inward = rotate90(sample.inward, center, turn);
      at_obj.push_back({at.x, at.y, 0});
      in_obj.push_back({inward.x, inward.y, 0});
    }
    const std::vector<cv::Point2d> at_img = project(pose, at_obj);
    const std::vector<cv::Point2d> in_img = project(pose, in_obj);
    std::vector<double> contrast(interior_.size(), 0);
    std::vector<char> visible(interior_.size(), 0);
    std::vector<double> pool;
    for (std::size_t i = 0; i < interior_.size(); ++i) {
      const float outer = gray_at(gray, at_img[i].x, at_img[i].y);
      const float inner = gray_at(gray, in_img[i].x, in_img[i].y);
      if (outer < 0.f || inner < 0.f) {
        continue;
      }
      contrast[i] = static_cast<double>(outer - inner);
      visible[i] = 1;
      pool.push_back(contrast[i]);
    }
    if (pool.size() < 8) {
      continue;
    }
    std::vector<double> sorted = pool;
    std::sort(sorted.begin(), sorted.end());
    const double scale = sorted[static_cast<std::size_t>(0.9 * static_cast<double>(sorted.size() - 1))];
    if (scale < params_.min_contrast) {
      continue;
    }
    struct Mark {
      double coord = 0;
      bool lit = false;
    };
    std::vector<std::vector<Mark>> marks(model_.runs.size());
    if (turn == 0) {
      bright_c.clear();
      dark_c.clear();
    }
    for (std::size_t i = 0; i < interior_.size(); ++i) {
      if (!visible[i]) {
        continue;
      }
      const Sample& sample = model_.samples[static_cast<std::size_t>(interior_[i])];
      const Run& run = model_.runs[static_cast<std::size_t>(sample.run)];
      const double value = contrast[i];
      if (turn == 0) {
        (run.bright ? bright_c : dark_c).push_back(value);
      }
      if (value > params_.lit_ratio * scale) {
        marks[static_cast<std::size_t>(sample.run)].push_back({sample.coord, true});
      } else if (value < params_.unlit_ratio * scale) {
        marks[static_cast<std::size_t>(sample.run)].push_back({sample.coord, false});
      }
    }
    for (std::size_t run_index = 0; run_index < model_.runs.size(); ++run_index) {
      const std::vector<Mark>& run_marks = marks[run_index];
      if (static_cast<int>(run_marks.size()) < params_.min_samples) {
        continue;
      }
      const bool expect_lit = model_.runs[run_index].bright;
      int agree = 0;
      int stretch = 0;
      int longest = 0;
      double previous = -1e9;
      for (const Mark& mark : run_marks) {
        const bool matches = mark.lit == expect_lit;
        if (matches) {
          ++agree;
        }
        // A dropped sample is off the edge. Do not join the two sides.
        if (mark.coord - previous > 1.5) {
          stretch = 0;
        }
        if (!matches) {
          ++stretch;
          longest = std::max(longest, stretch);
        } else {
          stretch = 0;
        }
        previous = mark.coord;
      }
      const int n = static_cast<int>(run_marks.size());
      const int disagree = n - agree;
      // A notch is a short dark block inside what a wrong turn treats as one
      // solid arm. Majority agreement hides it. Once the pose sits on the
      // border, a contiguous opposing block of min_samples is a contradiction.
      const bool notch = notch_block && longest >= params_.min_samples;
      bool confirmed_run = false;
      if (disagree >= params_.agree_frac * n || notch) {
        ++result.contradicted[turn];
      } else if (agree >= params_.agree_frac * n) {
        ++result.confirmed[turn];
        confirmed_run = true;
      }
      const Run& run = model_.runs[run_index];
      const bool gap = turn == 0 && !run.bright && std::abs((run.b - run.a) - model_.notch_gap) < 1e-3;
      if (gap) {
        ++result.gap_present[0];
        if (confirmed_run) {
          ++result.gap_confirmed[0];
        }
      }
    }
    result.feasible[turn] = result.contradicted[turn] == 0 && result.confirmed[turn] >= params_.min_confirmed_runs;
  }
  result.separation = median_of(bright_c) - median_of(dark_c);
  return result;
}

double MarkerDetector::edge_offset(const Pose& pose, const EdgePoint& edge, const cv::Mat& gray,
                                   bool& on_screen) const {
  on_screen = false;
  const std::vector<cv::Point2d> image =
      project(pose, {{edge.p.x, edge.p.y, 0}, {edge.p.x + edge.n.x, edge.p.y + edge.n.y, 0}});
  if (image.size() != 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  cv::Point2d direction = image[1] - image[0];
  const double pixels_per_unit = std::hypot(direction.x, direction.y);
  if (!(pixels_per_unit > 1e-4)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  direction *= 1.0 / pixels_per_unit;
  // The inner edge of the arm is one frame width inward and has the opposite
  // sign, so the window can be a whole stroke. The sign keeps that edge out.
  // Clipping the samples to the image, rather than requiring the whole window
  // to fit, keeps a border that sits near the frame.
  const double stroke = pixels_per_unit * model_.frame;
  const double radius = std::max(params_.min_search_px, params_.search_widths * stroke);
  if (radius < 1.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const cv::Point2d origin = image[0];
  if (origin.x < 2 || origin.y < 2 || origin.x >= gray.cols - 2 || origin.y >= gray.rows - 2) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  on_screen = true;
  auto contrast_at = [&](double t) {
    const float inner = gray_at(gray, origin.x + (t - 1.0) * direction.x, origin.y + (t - 1.0) * direction.y);
    const float outer = gray_at(gray, origin.x + (t + 1.0) * direction.x, origin.y + (t + 1.0) * direction.y);
    if (inner < 0.f || outer < 0.f) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(inner - outer);
  };
  // Nearest local peak, not the strongest. A rail or the far side of a wide
  // arm can outscore the border that this sample belongs to.
  double best_t = std::numeric_limits<double>::quiet_NaN();
  double best_abs = std::numeric_limits<double>::infinity();
  for (double t = -radius; t <= radius + 1e-9; t += 0.5) {
    const double contrast = contrast_at(t);
    if (!std::isfinite(contrast) || contrast < params_.min_contrast) {
      continue;
    }
    const double previous = contrast_at(t - 0.5);
    const double next = contrast_at(t + 0.5);
    if (std::isfinite(previous) && contrast < previous) {
      continue;
    }
    if (std::isfinite(next) && contrast < next) {
      continue;
    }
    if (std::abs(t) < best_abs) {
      best_abs = std::abs(t);
      best_t = t;
    }
  }
  if (!std::isfinite(best_t)) {
    return best_t;
  }
  const double left = contrast_at(best_t - 0.5);
  const double mid = contrast_at(best_t);
  const double right = contrast_at(best_t + 0.5);
  const double denom = left - 2.0 * mid + right;
  if (std::isfinite(left) && std::isfinite(mid) && std::isfinite(right) && std::abs(denom) > 1e-6) {
    const double delta = 0.5 * (left - right) / (2.0 * denom);
    if (std::abs(delta) <= 0.5) {
      best_t += delta;
    }
  }
  return best_t;
}

MarkerDetector::EdgeFit MarkerDetector::fit_edges(const Pose& pose, const cv::Mat& gray) const {
  EdgeFit fit;
  int inliers = 0;
  int side_visible[4] = {};
  int side_found[4] = {};
  int side_inliers[4] = {};
  double side_lo[4];
  double side_hi[4];
  for (int side = 0; side < 4; ++side) {
    side_lo[side] = std::numeric_limits<double>::infinity();
    side_hi[side] = -std::numeric_limits<double>::infinity();
  }
  for (int i = 0; i < static_cast<int>(edges_.size()); ++i) {
    bool on_screen = false;
    const double value = edge_offset(pose, edges_[static_cast<std::size_t>(i)], gray, on_screen);
    const int side = edges_[static_cast<std::size_t>(i)].side;
    if (on_screen) {
      ++fit.visible;
      ++side_visible[side];
    }
    if (!std::isfinite(value)) {
      continue;
    }
    fit.index.push_back(i);
    fit.residual.push_back(value);
    ++side_found[side];
    const double magnitude = std::abs(value);
    const double weight = magnitude <= params_.huber_px ? 1.0 : params_.huber_px / magnitude;
    fit.cost += weight * value * value;
    if (magnitude <= params_.huber_px) {
      ++inliers;
      ++side_inliers[side];
      const EdgePoint& edge = edges_[static_cast<std::size_t>(i)];
      const double coord = (side == 0 || side == 2) ? edge.p.x : edge.p.y;
      side_lo[side] = std::min(side_lo[side], coord);
      side_hi[side] = std::max(side_hi[side], coord);
    }
  }
  const int n = static_cast<int>(fit.index.size());
  // A known square's pose is fixed by three line correspondences. Hits piled
  // on one side leave the scale free, and the other three corners are then
  // an extrapolation that changes every time the visible sliver changes.
  int observed_sides = 0;
  int spanned_sides = 0;
  for (int side = 0; side < 4; ++side) {
    if (side_visible[side] < params_.min_samples || side_found[side] == 0) {
      continue;
    }
    const bool covered =
        static_cast<double>(side_found[side]) >= params_.min_edge_frac * static_cast<double>(side_visible[side]);
    const bool agreed =
        static_cast<double>(side_inliers[side]) >= params_.agree_frac * static_cast<double>(side_found[side]);
    // The far corner of a line sits about one side beyond a short segment.
    // Inliers have to span half the side or that corner is an extrapolation.
    const bool spanned = side_hi[side] - side_lo[side] >= params_.min_side_span * model_.outer;
    if (covered && agreed) {
      ++observed_sides;
      if (spanned) {
        ++spanned_sides;
      }
    }
  }
  fit.enough = fit.visible > 0 && n >= params_.min_edge_samples &&
               static_cast<double>(n) >= params_.min_edge_frac * static_cast<double>(fit.visible);
  fit.locked = fit.enough && observed_sides >= 3 &&
               static_cast<double>(inliers) >= params_.agree_frac * static_cast<double>(n);
  fit.determined = fit.locked && spanned_sides >= 3;
  return fit;
}

MarkerDetector::Pose MarkerDetector::refine(const Pose& pose, const cv::Mat& gray) const {
  Pose current = pose;
  EdgeFit best = fit_edges(current, gray);
  if (!best.enough) {
    return pose;
  }
  double lambda = 1e-2;
  const double step_eps[6] = {1e-4, 1e-4, 1e-4, 0.05, 0.05, 0.05};
  for (int iter = 0; iter < params_.lm_iters; ++iter) {
    const int rows = static_cast<int>(best.index.size());
    cv::Mat jacobian(rows, 6, CV_64F);
    cv::Mat residual(rows, 1, CV_64F);
    for (int r = 0; r < rows; ++r) {
      residual.at<double>(r, 0) = best.residual[static_cast<std::size_t>(r)];
      const double magnitude = std::abs(residual.at<double>(r, 0));
      const double weight = magnitude <= params_.huber_px ? 1.0 : params_.huber_px / magnitude;
      const double scale = std::sqrt(weight);
      residual.at<double>(r, 0) *= scale;
      const EdgePoint& edge = edges_[static_cast<std::size_t>(best.index[static_cast<std::size_t>(r)])];
      const std::vector<cv::Point2d> axis =
          project(current, {{edge.p.x, edge.p.y, 0}, {edge.p.x + edge.n.x, edge.p.y + edge.n.y, 0}});
      cv::Point2d normal(1, 0);
      if (axis.size() == 2) {
        normal = axis[1] - axis[0];
        const double length = std::hypot(normal.x, normal.y);
        if (length > 1e-4) {
          normal *= 1.0 / length;
        }
      }
      // Moving the sample along the outward normal decreases the measured
      // offset, so the pose derivative is the negative of that image motion.
      // Differentiating the projection, not the half-pixel search.
      auto along = [&](const Pose& trial) {
        const std::vector<cv::Point2d> moved = project(trial, {{edge.p.x, edge.p.y, 0}});
        if (moved.size() != 1) {
          return std::numeric_limits<double>::quiet_NaN();
        }
        return normal.x * moved[0].x + normal.y * moved[0].y;
      };
      for (int c = 0; c < 6; ++c) {
        Pose plus = current;
        Pose minus = current;
        if (c < 3) {
          plus.r[c] += step_eps[c];
          minus.r[c] -= step_eps[c];
        } else {
          plus.t[c - 3] += step_eps[c];
          minus.t[c - 3] -= step_eps[c];
        }
        const double high = along(plus);
        const double low = along(minus);
        if (!std::isfinite(high) || !std::isfinite(low)) {
          jacobian.at<double>(r, c) = 0;
        } else {
          jacobian.at<double>(r, c) = scale * (-(high - low) / (2.0 * step_eps[c]));
        }
      }
    }
    cv::Mat normal = jacobian.t() * jacobian;
    cv::Mat right = jacobian.t() * residual;
    for (int c = 0; c < 6; ++c) {
      normal.at<double>(c, c) *= 1.0 + lambda;
    }
    cv::Mat delta;
    if (!cv::solve(normal, right, delta, cv::DECOMP_SVD) || delta.rows != 6) {
      lambda *= 10.0;
      continue;
    }
    Pose trial = current;
    bool finite = true;
    for (int c = 0; c < 6; ++c) {
      const double value = delta.at<double>(c, 0);
      if (!std::isfinite(value)) {
        finite = false;
        break;
      }
      if (c < 3) {
        trial.r[c] -= value;
      } else {
        trial.t[c - 3] -= value;
      }
    }
    if (!finite || !facing(trial)) {
      lambda *= 10.0;
      continue;
    }
    const std::array<cv::Point2d, 4> before = plate(current);
    const std::array<cv::Point2d, 4> after = plate(trial);
    double shift = 0;
    double side = 0;
    for (int i = 0; i < 4; ++i) {
      shift += cv::norm(after[static_cast<std::size_t>(i)] - before[static_cast<std::size_t>(i)]);
      side += cv::norm(before[static_cast<std::size_t>((i + 1) % 4)] - before[static_cast<std::size_t>(i)]);
    }
    if (shift > 0.5 * side) {
      lambda *= 10.0;
      continue;
    }
    EdgeFit next = fit_edges(trial, gray);
    if (next.enough && next.cost < best.cost) {
      current = trial;
      best = std::move(next);
      lambda = std::max(1e-6, lambda * 0.3);
    } else {
      lambda *= 10.0;
    }
    if (lambda > 1e6) {
      break;
    }
  }
  current.valid = true;
  return current;
}

bool MarkerDetector::cold_start(const cv::Mat& gray, Pose& out) const {
  cv::Mat work;
  const double scale = params_.segment_scale;
  if (scale < 0.999) {
    cv::resize(gray, work, cv::Size(), scale, scale, cv::INTER_AREA);
  } else {
    work = gray;
  }
  const double inv = 1.0 / scale;
  struct Shape {
    std::vector<cv::Point2d> poly;
    double area = 0;
  };
  std::map<std::pair<int, int>, Shape> cells;
  std::vector<int> levels = params_.levels;
  std::sort(levels.begin(), levels.end());
  for (const int level : levels) {
    cv::Mat binary;
    cv::threshold(work, binary, level, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
      if (contour.size() < 4) {
        continue;
      }
      std::vector<cv::Point> approx;
      cv::approxPolyDP(contour, approx, params_.approx_frac * cv::arcLength(contour, true), true);
      if (approx.size() != 6) {
        continue;
      }
      const double area = std::abs(cv::contourArea(approx)) * inv * inv;
      if (area < params_.min_area) {
        continue;
      }
      cv::Point2d centroid(0, 0);
      std::vector<cv::Point2d> poly;
      poly.reserve(6);
      for (const cv::Point& point : approx) {
        const cv::Point2d full(point.x * inv, point.y * inv);
        poly.push_back(full);
        centroid += full;
      }
      centroid *= 1.0 / 6.0;
      const auto key = std::make_pair(static_cast<int>(std::floor(centroid.x / params_.dedup_px)),
                                      static_cast<int>(std::floor(centroid.y / params_.dedup_px)));
      const auto found = cells.find(key);
      if (found != cells.end() && found->second.area >= area) {
        continue;
      }
      cells[key] = Shape{std::move(poly), area};
    }
  }
  std::vector<Shape> shapes;
  shapes.reserve(cells.size());
  for (auto& entry : cells) {
    shapes.push_back(std::move(entry.second));
  }
  if (static_cast<int>(shapes.size()) > params_.max_polygons) {
    std::sort(shapes.begin(), shapes.end(),
              [](const Shape& a, const Shape& b) { return a.area < b.area; });
    std::vector<Shape> kept;
    const double stride = static_cast<double>(shapes.size()) / static_cast<double>(params_.max_polygons);
    for (int i = 0; i < params_.max_polygons; ++i) {
      kept.push_back(std::move(shapes[static_cast<std::size_t>(i * stride)]));
    }
    shapes = std::move(kept);
  }

  struct Hypothesis {
    Pose pose;
    double rms = 0;
    double separation = 0;
    int confirmed = 0;
  };
  std::vector<Hypothesis> ranked;
  for (const Shape& shape : shapes) {
    for (const auto& seed : model_.seeds) {
      for (int start = 0; start < 6; ++start) {
        for (const int direction : {1, -1}) {
          std::vector<cv::Point2f> object2(6);
          std::vector<cv::Point2f> image2(6);
          std::vector<cv::Point3d> object3(6);
          std::vector<cv::Point2d> image3(6);
          for (int i = 0; i < 6; ++i) {
            int k = start + direction * i;
            k %= 6;
            if (k < 0) {
              k += 6;
            }
            object2[static_cast<std::size_t>(i)] =
                cv::Point2f(static_cast<float>(seed[static_cast<std::size_t>(i)].x),
                            static_cast<float>(seed[static_cast<std::size_t>(i)].y));
            image2[static_cast<std::size_t>(i)] =
                cv::Point2f(static_cast<float>(shape.poly[static_cast<std::size_t>(k)].x),
                            static_cast<float>(shape.poly[static_cast<std::size_t>(k)].y));
            object3[static_cast<std::size_t>(i)] = {seed[static_cast<std::size_t>(i)].x,
                                                    seed[static_cast<std::size_t>(i)].y, 0};
            image3[static_cast<std::size_t>(i)] = shape.poly[static_cast<std::size_t>(k)];
          }
          const cv::Mat homography = cv::findHomography(object2, image2, 0);
          if (homography.empty()) {
            continue;
          }
          std::vector<cv::Point2f> predicted;
          cv::perspectiveTransform(object2, predicted, homography);
          double square = 0;
          cv::Point2d centroid(0, 0);
          for (int i = 0; i < 6; ++i) {
            centroid += image3[static_cast<std::size_t>(i)];
          }
          centroid *= 1.0 / 6.0;
          double radius = 0;
          for (int i = 0; i < 6; ++i) {
            const cv::Point2f delta = predicted[static_cast<std::size_t>(i)] - image2[static_cast<std::size_t>(i)];
            square += static_cast<double>(delta.x) * delta.x + static_cast<double>(delta.y) * delta.y;
            radius += cv::norm(image3[static_cast<std::size_t>(i)] - centroid);
          }
          radius /= 6.0;
          const double rms = std::sqrt(square / 6.0);
          if (!(radius > 1.0) || rms > params_.reproj_frac * radius) {
            continue;
          }
          std::vector<cv::Mat> rvecs;
          std::vector<cv::Mat> tvecs;
          const int solutions =
              cv::solvePnPGeneric(object3, image3, camera_, dist_, rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
          for (int s = 0; s < solutions; ++s) {
            cv::Mat r64;
            cv::Mat t64;
            rvecs[static_cast<std::size_t>(s)].reshape(1, 3).convertTo(r64, CV_64F);
            tvecs[static_cast<std::size_t>(s)].reshape(1, 3).convertTo(t64, CV_64F);
            Hypothesis hypothesis;
            hypothesis.pose.r = {r64.at<double>(0), r64.at<double>(1), r64.at<double>(2)};
            hypothesis.pose.t = {t64.at<double>(0), t64.at<double>(1), t64.at<double>(2)};
            hypothesis.pose.valid = true;
            hypothesis.rms = rms;
            if (!facing(hypothesis.pose)) {
              continue;
            }
            ranked.push_back(hypothesis);
          }
        }
      }
    }
  }
  if (ranked.empty()) {
    return false;
  }
  std::sort(ranked.begin(), ranked.end(),
            [](const Hypothesis& a, const Hypothesis& b) { return a.rms < b.rms; });
  if (ranked.size() > 30) {
    ranked.resize(30);
  }
  // The contour is the bloom, so a coarse pose can put a few runs on the
  // wrong side of a boundary. Keep a pose when turn 0 is the best reading
  // and has real contrast. The notch has to be unique after the edge lock.
  std::vector<Hypothesis> unique;
  for (Hypothesis& hypothesis : ranked) {
    const Orientation orientation = orient(hypothesis.pose, gray, false);
    const int margin0 = orientation.confirmed[0] - orientation.contradicted[0];
    bool best_turn = orientation.confirmed[0] >= params_.min_confirmed_runs &&
                     orientation.separation >= params_.min_contrast;
    for (int turn = 1; best_turn && turn < 4; ++turn) {
      const int margin = orientation.confirmed[turn] - orientation.contradicted[turn];
      if (margin >= margin0) {
        best_turn = false;
      }
    }
    if (!best_turn) {
      continue;
    }
    hypothesis.separation = orientation.separation;
    hypothesis.confirmed = orientation.confirmed[0];
    unique.push_back(hypothesis);
  }
  if (unique.empty()) {
    return false;
  }
  std::sort(unique.begin(), unique.end(), [](const Hypothesis& a, const Hypothesis& b) {
    if (a.confirmed != b.confirmed) {
      return a.confirmed > b.confirmed;
    }
    return a.separation > b.separation;
  });
  if (unique.size() > 8) {
    unique.resize(8);
  }
  // Two planar solutions of one L reproject that L together and the outer
  // corners apart. Lock each survivor to the whole border, then keep a pose
  // only when no other locked pose is a different plate.
  std::vector<Hypothesis> locked;
  for (const Hypothesis& hypothesis : unique) {
    const std::array<cv::Point2d, 4> guess = plate(hypothesis.pose);
    double guess_side = 0;
    for (int i = 0; i < 4; ++i) {
      guess_side += cv::norm(guess[static_cast<std::size_t>((i + 1) % 4)] - guess[static_cast<std::size_t>(i)]);
    }
    guess_side /= 4.0;
    bool already = false;
    for (const Hypothesis& kept : locked) {
      const std::array<cv::Point2d, 4> other = plate(kept.pose);
      double gap = 0;
      for (int k = 0; k < 4; ++k) {
        gap += cv::norm(other[static_cast<std::size_t>(k)] - guess[static_cast<std::size_t>(k)]);
      }
      if (guess_side > 1.0 && gap / 4.0 <= params_.pose_agree_frac * guess_side) {
        already = true;
        break;
      }
    }
    if (already) {
      continue;
    }
    Pose refined = refine(hypothesis.pose, gray);
    Orientation orientation = orient(refined, gray, true);
    const EdgeFit fit = fit_edges(refined, gray);
    if (!orientation.unique() || !facing(refined) || !fit.determined) {
      continue;
    }
    Hypothesis kept = hypothesis;
    kept.pose = refined;
    kept.separation = orientation.separation;
    kept.confirmed = orientation.confirmed[0];
    locked.push_back(kept);
  }
  if (locked.empty()) {
    return false;
  }
  double best_mean = std::numeric_limits<double>::infinity();
  std::vector<double> mean_cost(locked.size(), 0);
  for (std::size_t i = 0; i < locked.size(); ++i) {
    const EdgeFit fit = fit_edges(locked[i].pose, gray);
    const double mean = fit.cost / static_cast<double>(std::max<int>(1, static_cast<int>(fit.residual.size())));
    mean_cost[i] = mean;
    best_mean = std::min(best_mean, mean);
  }
  std::vector<Hypothesis> competitive;
  for (std::size_t i = 0; i < locked.size(); ++i) {
    if (mean_cost[i] <= params_.rival_cost_ratio * best_mean) {
      competitive.push_back(locked[i]);
    }
  }
  const std::array<cv::Point2d, 4> reference = plate(competitive.front().pose);
  double side = 0;
  for (int i = 0; i < 4; ++i) {
    side += cv::norm(reference[static_cast<std::size_t>((i + 1) % 4)] - reference[static_cast<std::size_t>(i)]);
  }
  side /= 4.0;
  for (std::size_t i = 1; i < competitive.size(); ++i) {
    const std::array<cv::Point2d, 4> other = plate(competitive[i].pose);
    double gap = 0;
    for (int k = 0; k < 4; ++k) {
      gap += cv::norm(other[static_cast<std::size_t>(k)] - reference[static_cast<std::size_t>(k)]);
    }
    if (!(side > 1.0) || gap / 4.0 > params_.pose_agree_frac * side) {
      return false;
    }
  }
  // One plate. The outer edge is already the positive contrast, so the
  // lock with the smaller residual is the border. A larger image square is
  // not a second ridge.
  double best_cost = std::numeric_limits<double>::infinity();
  bool any = false;
  for (std::size_t i = 0; i < locked.size(); ++i) {
    if (mean_cost[i] > params_.rival_cost_ratio * best_mean) {
      continue;
    }
    if (mean_cost[i] < best_cost) {
      best_cost = mean_cost[i];
      out = locked[i].pose;
      any = true;
    }
  }
  if (!any) {
    return false;
  }
  out.valid = true;
  return true;
}

Detection MarkerDetector::publish(const Pose& pose, double score) const {
  Detection detection;
  const std::array<cv::Point2d, 4> corners = plate(pose);
  for (int i = 0; i < 4; ++i) {
    const cv::Point2d& corner = corners[static_cast<std::size_t>(i)];
    if (!std::isfinite(corner.x) || !std::isfinite(corner.y)) {
      return {};
    }
    detection.corners[static_cast<std::size_t>(i)] =
        cv::Point2f(static_cast<float>(corner.x), static_cast<float>(corner.y));
    detection.measured[static_cast<std::size_t>(i)] = true;
  }
  detection.found = true;
  detection.support = 4;
  detection.score = static_cast<float>(score);
  detection.notch = model_.notch_corner;
  return detection;
}

MarkerDetector::Pose MarkerDetector::predict() const {
  Pose pose = pose_;
  pose.r += vel_r_;
  pose.t += vel_t_;
  return pose;
}

bool MarkerDetector::within_basin(const Pose& seed, const Pose& refined) const {
  const std::array<cv::Point2d, 4> before = plate(seed);
  const std::array<cv::Point2d, 4> after = plate(refined);
  double side = 0;
  double gap = 0;
  for (int i = 0; i < 4; ++i) {
    side += cv::norm(before[static_cast<std::size_t>((i + 1) % 4)] - before[static_cast<std::size_t>(i)]);
    gap += cv::norm(after[static_cast<std::size_t>(i)] - before[static_cast<std::size_t>(i)]);
  }
  side /= 4.0;
  gap /= 4.0;
  if (!(side > 1.0) || !(model_.outer > 0.0)) {
    return false;
  }
  // The normal search reaches one stroke (search_widths of the frame). The
  // inner edge is the opposite sign, so a step inside this window is still
  // the outer border. A step past the window was not this edge.
  const double stroke = side * model_.frame / model_.outer;
  return gap < params_.search_widths * stroke;
}

bool MarkerDetector::code_accepts(const Orientation& code, bool acquiring) const {
  for (int turn = 1; turn < 4; ++turn) {
    if (code.feasible[turn]) {
      return false;
    }
  }
  if (code.contradicted[0] != 0) {
    return false;
  }
  // A new track has to read the notch. A track already identified keeps the
  // border when this frame's read is only short, and drops it when the read
  // contradicts that notch.
  if (!acquiring && identity_) {
    return true;
  }
  return code.unique();
}

void MarkerDetector::remember(const Pose& next, bool adopt_velocity) {
  if (adopt_velocity && pose_.valid) {
    vel_r_ = next.r - pose_.r;
    vel_t_ = next.t - pose_.t;
  } else {
    vel_r_ = cv::Vec3d();
    vel_t_ = cv::Vec3d();
  }
  pose_ = next;
  pose_.valid = true;
}

void MarkerDetector::clear_track() {
  pose_ = Pose();
  vel_r_ = cv::Vec3d();
  vel_t_ = cv::Vec3d();
  identity_ = false;
}

Detection MarkerDetector::detect(const cv::Mat& bgr) {
  if (!ready_ || bgr.empty() || bgr.depth() != CV_8U) {
    clear_track();
    return {};
  }
  cv::Mat gray;
  if (bgr.channels() == 3) {
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  } else if (bgr.channels() == 1) {
    gray = bgr;
  } else {
    clear_track();
    return {};
  }
  try {
    if (pose_.valid) {
      const Pose seed = predict();
      const Pose refined = refine(seed, gray);
      const Orientation orientation = orient(refined, gray, true);
      const EdgeFit fit = fit_edges(refined, gray);
      const bool border = facing(refined) && fit.locked && within_basin(seed, refined);
      if (border && code_accepts(orientation, false)) {
        if (orientation.unique()) {
          identity_ = true;
        }
        remember(refined, true);
        return publish(refined, orientation.separation);
      }
      // The seed is not this marker on this frame. Forget it and read the
      // frame on its own. The seed must not erase a reading.
      clear_track();
    }
    Pose found;
    if (!cold_start(gray, found)) {
      return {};
    }
    identity_ = true;
    remember(found, false);
    return publish(found, orient(found, gray, true).separation);
  } catch (const cv::Exception&) {
    clear_track();
    return {};
  }
}

void draw_detection(cv::Mat& bgr, const Detection& detection) {
  const cv::Scalar colors[4] = {{255, 196, 64}, {80, 220, 255}, {80, 80, 255}, {96, 220, 96}};
  const char* names[4] = {"LT", "RT", "RB", "LB"};
  const int font = cv::FONT_HERSHEY_SIMPLEX;
  auto shade = [&](cv::Rect plate) {
    plate &= cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (plate.empty() || bgr.channels() != 3) {
      return;
    }
    cv::Mat roi = bgr(plate);
    cv::Mat dark(roi.size(), roi.type(), cv::Scalar(16, 16, 16));
    cv::addWeighted(roi, 0.30, dark, 0.70, 0, roi);
    cv::rectangle(bgr, plate, {230, 230, 230}, 1, cv::LINE_AA);
  };
  auto text_at = [&](const std::string& text, cv::Point origin, cv::Scalar color, double scale) {
    cv::putText(bgr, text, origin, font, scale, {0, 0, 0}, 3, cv::LINE_AA);
    cv::putText(bgr, text, origin, font, scale, color, 1, cv::LINE_AA);
  };

  if (!detection.found) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize("undetected", font, 0.7, 1, &baseline);
    const int pad = 12;
    const cv::Rect plate(20, 20, size.width + pad * 2, size.height + baseline + pad * 2);
    shade(plate);
    text_at("undetected", {plate.x + pad, plate.y + pad + size.height}, {240, 240, 240}, 0.7);
    return;
  }

  const cv::Rect bounds(0, 0, bgr.cols, bgr.rows);
  std::array<cv::Point, 4> clipped_a{};
  std::array<cv::Point, 4> clipped_b{};
  std::array<bool, 4> drawn{};
  for (int i = 0; i < 4; ++i) {
    cv::Point a(cvRound(detection.corners[static_cast<std::size_t>(i)].x),
                cvRound(detection.corners[static_cast<std::size_t>(i)].y));
    cv::Point b(cvRound(detection.corners[static_cast<std::size_t>((i + 1) % 4)].x),
                cvRound(detection.corners[static_cast<std::size_t>((i + 1) % 4)].y));
    drawn[static_cast<std::size_t>(i)] = cv::clipLine(bounds, a, b);
    clipped_a[static_cast<std::size_t>(i)] = a;
    clipped_b[static_cast<std::size_t>(i)] = b;
  }
  for (int i = 0; i < 4; ++i) {
    if (drawn[static_cast<std::size_t>(i)]) {
      cv::line(bgr, clipped_a[static_cast<std::size_t>(i)], clipped_b[static_cast<std::size_t>(i)], {0, 0, 0}, 5,
               cv::LINE_AA);
    }
  }
  for (int i = 0; i < 4; ++i) {
    if (drawn[static_cast<std::size_t>(i)]) {
      cv::line(bgr, clipped_a[static_cast<std::size_t>(i)], clipped_b[static_cast<std::size_t>(i)], {255, 255, 255}, 2,
               cv::LINE_AA);
    }
  }
  for (int i = 0; i < 4; ++i) {
    const cv::Point2f corner = detection.corners[static_cast<std::size_t>(i)];
    if (corner.x < 0.f || corner.y < 0.f || corner.x >= static_cast<float>(bgr.cols) ||
        corner.y >= static_cast<float>(bgr.rows)) {
      continue;
    }
    const cv::Point center(cvRound(corner.x), cvRound(corner.y));
    cv::circle(bgr, center, 8, {0, 0, 0}, 3, cv::LINE_AA);
    cv::circle(bgr, center, 8, colors[i], 1, cv::LINE_AA);
  }

  // Names stay in one plate, in plate order, so they never sit on the lamp.
  // A hollow swatch is a corner that falls outside this image.
  const int pad = 14;
  const int col_w = 112;
  const int row_h = 34;
  const int caption = 26;
  const cv::Rect plate(20, 20, pad * 2 + col_w * 2, pad * 2 + row_h * 2 + caption);
  shade(plate);
  const int visual[4] = {0, 1, 3, 2};
  for (int cell = 0; cell < 4; ++cell) {
    const int index = visual[cell];
    const int col = cell % 2;
    const int row = cell / 2;
    const int x = plate.x + pad + col * col_w;
    const int y = plate.y + pad + row * row_h;
    const cv::Point2f corner = detection.corners[static_cast<std::size_t>(index)];
    const bool inside = corner.x >= 0.f && corner.y >= 0.f && corner.x < static_cast<float>(bgr.cols) &&
                        corner.y < static_cast<float>(bgr.rows);
    const cv::Rect swatch(x, y + (row_h - 14) / 2, 14, 14);
    const cv::Scalar ink = inside ? colors[index] : cv::Scalar(150, 150, 150);
    if (inside) {
      cv::rectangle(bgr, swatch, colors[index], cv::FILLED, cv::LINE_AA);
    } else {
      cv::rectangle(bgr, swatch, ink, 1, cv::LINE_AA);
    }
    int baseline = 0;
    const cv::Size size = cv::getTextSize(names[index], font, 0.6, 1, &baseline);
    text_at(names[index], {swatch.x + swatch.width + 8, y + (row_h + size.height) / 2 - 1},
            inside ? cv::Scalar(245, 245, 245) : ink, 0.6);
  }
  if (detection.notch >= 0 && detection.notch < 4) {
    const std::string caption_text = std::string("notch ") + names[detection.notch];
    text_at(caption_text, {plate.x + pad, plate.y + plate.height - pad + 2}, colors[detection.notch], 0.5);
  }
}
