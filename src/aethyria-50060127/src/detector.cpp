#include "detector.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <vector>

namespace {

// Labeled drawing sizes, in drawing units. The plate is 80, each corner cell is 30,
// and the arm inner edge is 22, so the arm is 8 thick. One model unit is the distance
// between adjacent solid-L centroids. A solid L of outer 30 and thickness 8 has area
// 416, and its centroid moment about the outer corner is 4304.
constexpr int kOuter = 80;
constexpr int kCell = 30;
constexpr int kArm = 8;
constexpr int kArea = kArm * (2 * kCell - kArm);
constexpr float kCx = 4304.f / static_cast<float>(kArea);
constexpr float kPitch = static_cast<float>(kOuter) - 2.f * kCx;
constexpr float kPlateLo = (0.f - kCx) / kPitch;
constexpr float kPlateHi = (static_cast<float>(kOuter) - kCx) / kPitch;
constexpr float kLSide = static_cast<float>(kCell) / kPitch;

// Detail A is that same L with two bites of width 8. The connected corner reaches 14;
// each arm keeps an 8-square from 22 to 30. The corner centroid lies on the diagonal
// of its own box. When no solid lamp verifies, the corner and one end fix a similarity,
// and the other end can tighten that similarity to an affine.
constexpr int kCornerOuter = 14;
constexpr float kCornerArea = static_cast<float>(kArm * (2 * kCornerOuter - kArm));
constexpr float kEndArea = static_cast<float>(kArm * kArm);

struct LampId {
  cv::Point2f at;
  cv::Point2f elbow;
};

const LampId kLamps[3] = {
    {{0.f, 0.f}, {-0.70710678f, -0.70710678f}},
    {{0.f, 1.f}, {-0.70710678f, 0.70710678f}},
    {{1.f, 1.f}, {0.70710678f, 0.70710678f}},
};

cv::Point2f to_model(float x, float y) { return {(x - kCx) / kPitch, (y - kCx) / kPitch}; }

cv::Point2f drawing_center(const cv::Point2f* pts, int n) {
  double twice = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  for (int i = 0; i < n; ++i) {
    const cv::Point2f& a = pts[i];
    const cv::Point2f& b = pts[(i + 1) % n];
    const double cross = static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
    twice += cross;
    sx += (static_cast<double>(a.x) + b.x) * cross;
    sy += (static_cast<double>(a.y) + b.y) * cross;
  }
  return {static_cast<float>(sx / (3.0 * twice)), static_cast<float>(sy / (3.0 * twice))};
}

cv::Point2f drawing_centroid(const cv::Point2f* pts, int n) {
  const cv::Point2f center = drawing_center(pts, n);
  return to_model(center.x, center.y);
}

struct PieceModel {
  cv::Point2f at;
  float area;
  cv::Point2f elbow;
};

const std::array<PieceModel, 3>& piece_models() {
  static const std::array<PieceModel, 3> models = [] {
    const cv::Point2f corner[] = {{80.f, 0.f}, {66.f, 0.f}, {66.f, 8.f}, {72.f, 8.f}, {72.f, 14.f}, {80.f, 14.f}};
    const cv::Point2f top[] = {{50.f, 0.f}, {58.f, 0.f}, {58.f, 8.f}, {50.f, 8.f}};
    const cv::Point2f side[] = {{72.f, 22.f}, {80.f, 22.f}, {80.f, 30.f}, {72.f, 30.f}};
    const cv::Point2f center = drawing_center(corner, 6);
    float min_x = corner[0].x;
    float max_x = corner[0].x;
    float min_y = corner[0].y;
    float max_y = corner[0].y;
    for (const cv::Point2f& point : corner) {
      min_x = std::min(min_x, point.x);
      max_x = std::max(max_x, point.x);
      min_y = std::min(min_y, point.y);
      max_y = std::max(max_y, point.y);
    }
    cv::Point2f elbow = center - cv::Point2f(0.5f * (min_x + max_x), 0.5f * (min_y + max_y));
    const float norm = std::hypot(elbow.x, elbow.y);
    elbow *= 1.f / norm;
    return std::array<PieceModel, 3>{
        PieceModel{drawing_centroid(corner, 6), kCornerArea, elbow},
        PieceModel{drawing_centroid(top, 4), kEndArea, {0.f, 0.f}},
        PieceModel{drawing_centroid(side, 4), kEndArea, {0.f, 0.f}},
    };
  }();
  return models;
}

struct Poly {
  cv::Point2f pts[6];
  int n;
};

const Poly kLampPolys[] = {
    {{{0.f, 0.f}, {30.f, 0.f}, {30.f, 8.f}, {8.f, 8.f}, {8.f, 30.f}, {0.f, 30.f}}, 6},
    {{{0.f, 80.f}, {30.f, 80.f}, {30.f, 72.f}, {8.f, 72.f}, {8.f, 50.f}, {0.f, 50.f}}, 6},
    {{{80.f, 80.f}, {80.f, 50.f}, {72.f, 50.f}, {72.f, 72.f}, {50.f, 72.f}, {50.f, 80.f}}, 6},
    {{{80.f, 0.f}, {66.f, 0.f}, {66.f, 8.f}, {72.f, 8.f}, {72.f, 14.f}, {80.f, 14.f}}, 6},
    {{{50.f, 0.f}, {58.f, 0.f}, {58.f, 8.f}, {50.f, 8.f}}, 4},
    {{{72.f, 22.f}, {80.f, 22.f}, {80.f, 30.f}, {72.f, 30.f}}, 4},
};

bool inside_poly(const cv::Point2f* pts, int n, float x, float y) {
  bool crossing = false;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    const float yi = pts[i].y;
    const float yj = pts[j].y;
    if ((yi > y) != (yj > y) && (x < (pts[j].x - pts[i].x) * (y - yi) / (yj - yi) + pts[i].x)) {
      crossing = !crossing;
    }
  }
  return crossing;
}

const std::vector<cv::Point2f>& lamp_samples() {
  static const std::vector<cv::Point2f> samples = [] {
    std::vector<cv::Point2f> out;
    for (int y = 1; y < kOuter; y += 2) {
      for (int x = 1; x < kOuter; x += 2) {
        for (const Poly& poly : kLampPolys) {
          if (inside_poly(poly.pts, poly.n, static_cast<float>(x), static_cast<float>(y))) {
            out.push_back(to_model(static_cast<float>(x), static_cast<float>(y)));
            break;
          }
        }
      }
    }
    return out;
  }();
  return samples;
}

// The plate center is dark for every 90° labeling. The two mouths belong only to the
// gapped corner, so they are scored on their own.
struct DarkSets {
  std::vector<cv::Point2f> center;
  std::vector<cv::Point2f> mouths;
};

const DarkSets& dark_sets() {
  static const DarkSets sets = [] {
    DarkSets built;
    auto add_rect = [&](std::vector<cv::Point2f>& out, int x0, int y0, int x1, int y1) {
      for (int y = y0; y < y1; y += 2) {
        for (int x = x0; x < x1; x += 2) {
          out.push_back(to_model(static_cast<float>(x), static_cast<float>(y)));
        }
      }
    };
    add_rect(built.center, 31, 31, 50, 50);
    add_rect(built.mouths, 59, 1, 66, 8);
    add_rect(built.mouths, 73, 15, 80, 22);
    return built;
  }();
  return sets;
}

cv::Point2f apply(const cv::Matx33d& h, cv::Point2f p) {
  const cv::Vec3d q = h * cv::Vec3d(p.x, p.y, 1.0);
  return cv::Point2f(static_cast<float>(q[0] / q[2]), static_cast<float>(q[1] / q[2]));
}

bool finite_point(cv::Point2f p) { return std::isfinite(p.x) && std::isfinite(p.y); }

bool finite_h(const cv::Matx33d& h) {
  for (int i = 0; i < 9; ++i) {
    if (!std::isfinite(h.val[i])) {
      return false;
    }
  }
  return true;
}

cv::Matx33d from_mat(const cv::Mat& m, int rows) {
  cv::Matx33d h = cv::Matx33d::eye();
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < m.cols; ++c) {
      h(r, c) = m.at<double>(r, c);
    }
  }
  return h;
}

float cross2(cv::Point2f o, cv::Point2f a, cv::Point2f b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

float imaged_pitch(const cv::Matx33d& h) {
  return static_cast<float>(cv::norm(apply(h, cv::Point2f(0.f, 1.f)) - apply(h, cv::Point2f(0.f, 0.f))));
}

bool plate_quad(const cv::Matx33d& h, std::array<cv::Point2f, 4>& quad) {
  const cv::Point2f model[4] = {
      {kPlateLo, kPlateLo},
      {kPlateHi, kPlateLo},
      {kPlateHi, kPlateHi},
      {kPlateLo, kPlateHi},
  };
  for (int i = 0; i < 4; ++i) {
    quad[i] = apply(h, model[i]);
    if (!finite_point(quad[i])) {
      return false;
    }
  }
  for (int i = 0; i < 4; ++i) {
    if (cross2(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]) <= 0.f) {
      return false;
    }
  }
  return true;
}

float elbow_dot(const cv::Matx33d& h, cv::Point2f model, cv::Point2f model_elbow, cv::Point2f observed) {
  const cv::Point2f predicted = apply(h, model + model_elbow * 0.25f) - apply(h, model);
  const double pn = cv::norm(predicted);
  const double on = cv::norm(observed);
  if (pn < 1e-4 || on < 1e-4) {
    return -1.f;
  }
  return static_cast<float>(predicted.dot(observed) / (pn * on));
}

bool elbow_agrees(const cv::Matx33d& h, cv::Point2f model, cv::Point2f model_elbow, cv::Point2f observed) {
  return elbow_dot(h, model, model_elbow, observed) > 0.50f;
}

// Zack's triangle. The cut is the histogram bin farthest under the line from the
// mode in [begin, end] to the last occupied bin. Called once on V of the whole
// frame, then on S of the pixels at or above that V, from the S mode toward 255.
int triangle_threshold(const std::vector<int>& hist, int begin, int end) {
  int peak = begin;
  for (int i = begin + 1; i <= end; ++i) {
    if (hist[i] > hist[peak]) {
      peak = i;
    }
  }
  int tail = end;
  while (tail > peak && hist[tail] == 0) {
    --tail;
  }
  if (tail <= peak + 1) {
    return peak;
  }
  const float dx = static_cast<float>(tail - peak);
  const float dy = static_cast<float>(hist[tail] - hist[peak]);
  const float norm = std::sqrt(dx * dx + dy * dy);
  int best = peak;
  float best_dist = -1.f;
  for (int i = peak; i <= tail; ++i) {
    const float dist = std::abs(dx * static_cast<float>(hist[i] - hist[peak]) - dy * static_cast<float>(i - peak)) / norm;
    if (dist > best_dist) {
      best_dist = dist;
      best = i;
    }
  }
  return best;
}

struct Score {
  float recall = 0.f;
  float coverage = 0.f;
  float center_ok = 0.f;
  float mouth_ok = 0.f;
  int center_in = 0;
  int mouth_in = 0;
  bool pass = false;
};

Score score_fit(const cv::Mat& hsv, const cv::Matx33d& h, int support, int v_cut, int s_cut) {
  Score score;
  const std::vector<cv::Point2f>& lamps = lamp_samples();
  int in_frame = 0;
  int bright = 0;
  for (cv::Point2f model : lamps) {
    const cv::Point2f p = apply(h, model);
    const int x = cvRound(p.x);
    const int y = cvRound(p.y);
    if (x < 0 || y < 0 || x >= hsv.cols || y >= hsv.rows) {
      continue;
    }
    ++in_frame;
    const cv::Vec3b pix = hsv.at<cv::Vec3b>(y, x);
    if (pix[2] >= v_cut && pix[1] <= s_cut) {
      ++bright;
    }
  }
  auto tally = [&](const std::vector<cv::Point2f>& samples, int& inside, int& dark_count) {
    for (cv::Point2f model : samples) {
      const cv::Point2f p = apply(h, model);
      const int x = cvRound(p.x);
      const int y = cvRound(p.y);
      if (x < 0 || y < 0 || x >= hsv.cols || y >= hsv.rows) {
        continue;
      }
      ++inside;
      if (hsv.at<cv::Vec3b>(y, x)[2] < v_cut) {
        ++dark_count;
      }
    }
  };
  int center_dark = 0;
  int mouth_dark = 0;
  const DarkSets& dark = dark_sets();
  tally(dark.center, score.center_in, center_dark);
  tally(dark.mouths, score.mouth_in, mouth_dark);
  score.coverage = lamps.empty() ? 0.f : static_cast<float>(in_frame) / static_cast<float>(lamps.size());
  if (in_frame < 12) {
    return score;
  }
  score.recall = static_cast<float>(bright) / static_cast<float>(in_frame);
  score.center_ok = score.center_in == 0 ? 0.f : static_cast<float>(center_dark) / static_cast<float>(score.center_in);
  score.mouth_ok = score.mouth_in == 0 ? 0.f : static_cast<float>(mouth_dark) / static_cast<float>(score.mouth_in);
  const bool mouth_pass = score.mouth_in < 24 || score.mouth_ok >= 0.50f;
  if (support <= 1) {
    score.pass = score.coverage >= 0.28f && score.recall >= 0.75f && score.center_in >= 12 && score.center_ok >= 0.75f &&
                 mouth_pass;
  } else {
    const bool center_pass = score.center_in < 8 || score.center_ok >= 0.70f;
    score.pass = score.coverage >= 0.20f && score.recall >= 0.55f && center_pass && mouth_pass;
  }
  return score;
}

std::array<cv::Point2f, 4> grow_quad(const std::array<cv::Point2f, 4>& quad, float scale) {
  cv::Point2f center(0.f, 0.f);
  for (const cv::Point2f& corner : quad) {
    center += corner;
  }
  center *= 0.25f;
  std::array<cv::Point2f, 4> grown{};
  for (int i = 0; i < 4; ++i) {
    grown[i] = center + (quad[i] - center) * scale;
  }
  return grown;
}

bool point_in_quad(const std::array<cv::Point2f, 4>& quad, cv::Point2f p) {
  for (int i = 0; i < 4; ++i) {
    if (cross2(quad[i], quad[(i + 1) % 4], p) < 0.f) {
      return false;
    }
  }
  return true;
}

struct Blob {
  cv::Point2f centroid;
  cv::Point2f elbow;
  cv::Point2f arm_x;
  cv::Point2f arm_y;
  cv::Point2f semantic[4]{};
  float area = 0.f;
  float solidity = 0.f;
  float extent = 0.f;
  float aspect = 0.f;
  float defect = 0.f;
  float mean_side = 0.f;
  bool is_l = false;
  bool is_piece = false;
  float quality = 0.f;
  std::vector<cv::Point2f> edge;
};

// A cell L inside the ordered plate has to land on a solid lamp. Blobs outside
// that region are other objects and do not reject the pose. Piece fragments are
// not in this list: their shape gate does not overlap the cell L.
bool explains_observed(const cv::Matx33d& h, const std::array<cv::Point2f, 4>& ordered, const std::vector<Blob>& ells) {
  const float pitch = imaged_pitch(h);
  if (!(pitch > 1.f)) {
    return false;
  }
  const std::array<cv::Point2f, 4> region = grow_quad(ordered, 1.20f);
  for (const Blob& blob : ells) {
    if (!point_in_quad(region, blob.centroid)) {
      continue;
    }
    float nearest = 1e9f;
    for (const LampId& lamp : kLamps) {
      nearest = std::min(nearest, static_cast<float>(cv::norm(apply(h, lamp.at) - blob.centroid)));
    }
    if (nearest > 0.20f * pitch) {
      return false;
    }
  }
  return true;
}

struct Fit {
  Detection detection;
  std::array<cv::Point2f, 4> measured{};
  float mouth_ok = 0.f;
  int mouth_in = 0;
  // Lamps were named from the previous plate, then refit on this frame.
  bool tracked = false;
  // Four outer edges of this labeling met as one homography.
  bool from_lines = false;
};

// A plate edge is the bright arm segments on one drawing line, separated by a dark
// gap. Lines stay lines under a homography, so four outer edges determine the
// plate together. Fewer than four leaves the lamp pose untouched: one edge is
// not allowed to move one corner.
std::array<cv::Point2f, 4> refine_plate(const cv::Matx33d& h, const std::array<cv::Point2f, 4>& quad,
                                        const std::vector<Blob>& ells, const std::vector<Blob>& pieces,
                                        const std::vector<Blob>& rims, int cols, int rows, bool& from_lines) {
  from_lines = false;
  const float pitch = imaged_pitch(h);
  if (!(pitch > 1.f) || cols < 3 || rows < 3) {
    return quad;
  }
  struct EdgeSeg {
    int side;
    float x0;
    float y0;
    float x1;
    float y1;
  };
  static const EdgeSeg kSegs[] = {
      {0, 0.f, 0.f, 30.f, 0.f},  {0, 50.f, 0.f, 58.f, 0.f},  {0, 66.f, 0.f, 80.f, 0.f},
      {1, 80.f, 0.f, 80.f, 14.f}, {1, 80.f, 22.f, 80.f, 30.f}, {1, 80.f, 50.f, 80.f, 80.f},
      {2, 0.f, 80.f, 30.f, 80.f}, {2, 50.f, 80.f, 80.f, 80.f}, {3, 0.f, 0.f, 0.f, 30.f},
      {3, 0.f, 50.f, 0.f, 80.f},
  };
  const float arm = static_cast<float>(kArm) / kPitch;
  const cv::Point2f inward[4] = {{0.f, arm}, {-arm, 0.f}, {0.f, -arm}, {arm, 0.f}};
  struct Candidate {
    cv::Point2f point;
    float signed_in;
  };
  std::array<std::vector<Candidate>, 4> candidates;
  std::array<cv::Point2f, 4> axis{};
  std::array<float, 4> axis_len{};
  std::array<float, 4> side_thickness{};
  constexpr float kAlong = 0.35f;

  for (const EdgeSeg& seg : kSegs) {
    const cv::Point2f model_a = to_model(seg.x0, seg.y0);
    const cv::Point2f model_b = to_model(seg.x1, seg.y1);
    const cv::Point2f a = apply(h, model_a);
    const cv::Point2f b = apply(h, model_b);
    if (!finite_point(a) || !finite_point(b)) {
      continue;
    }
    const cv::Point2f ab = b - a;
    const float ab_len = static_cast<float>(cv::norm(ab));
    if (ab_len < 1.f) {
      continue;
    }
    const cv::Point2f mid = (model_a + model_b) * 0.5f;
    const cv::Point2f dir = ab * (1.f / ab_len);
    cv::Point2f normal(-dir.y, dir.x);
    const cv::Point2f inward_img = apply(h, mid + inward[seg.side]) - apply(h, mid);
    if (normal.dot(inward_img) < 0.f) {
      normal = -normal;
    }
    const float thickness = normal.dot(inward_img);
    if (!(thickness > 1.f)) {
      continue;
    }
    const int side = seg.side;
    if (ab_len > axis_len[static_cast<size_t>(side)]) {
      axis[static_cast<size_t>(side)] = dir;
      axis_len[static_cast<size_t>(side)] = ab_len;
      side_thickness[static_cast<size_t>(side)] = thickness;
    }
    const float wide = std::max(1.20f * thickness, 0.70f * pitch);
    const float ab2 = ab.dot(ab);
    const float x_max = static_cast<float>(cols - 2);
    const float y_max = static_cast<float>(rows - 2);
    const cv::Point2f mid_img = (a + b) * 0.5f;
    const float near_blob = 1.20f * pitch;
    auto take = [&](const std::vector<Blob>& blobs) {
      for (const Blob& blob : blobs) {
        if (static_cast<float>(cv::norm(blob.centroid - mid_img)) > near_blob) {
          continue;
        }
        for (const cv::Point2f& point : blob.edge) {
          if (point.x < 1.f || point.y < 1.f || point.x > x_max || point.y > y_max) {
            continue;
          }
          const float t = (point - a).dot(ab) / ab2;
          if (t < -kAlong || t > 1.f + kAlong) {
            continue;
          }
          const float signed_in = (point - a).dot(normal);
          if (signed_in < -wide || signed_in > wide) {
            continue;
          }
          candidates[static_cast<size_t>(side)].push_back(Candidate{point, signed_in});
        }
      }
    };
    take(ells);
    take(pieces);
    take(rims);
  }

  std::array<std::vector<cv::Point2f>, 4> samples;
  for (int side = 0; side < 4; ++side) {
    std::vector<Candidate>& pool = candidates[static_cast<size_t>(side)];
    const float thickness = side_thickness[static_cast<size_t>(side)];
    if (pool.size() < 12 || !(thickness > 1.f)) {
      continue;
    }
    std::sort(pool.begin(), pool.end(), [](const Candidate& lhs, const Candidate& rhs) {
      return lhs.signed_in < rhs.signed_in;
    });
    const float ref = pool[pool.size() / 5].signed_in;
    if (ref > 0.55f * thickness || ref < -0.75f * pitch) {
      continue;
    }
    const float outer = ref - 0.10f * thickness;
    const float inner = ref + 0.35f * thickness;
    std::vector<cv::Point2f>& kept = samples[static_cast<size_t>(side)];
    kept.reserve(pool.size());
    for (const Candidate& cand : pool) {
      if (cand.signed_in >= outer && cand.signed_in <= inner) {
        kept.push_back(cand.point);
      }
    }
  }

  struct SideLine {
    bool ok = false;
    cv::Point2f origin;
    cv::Point2f direction;
    float thickness = 0.f;
  };
  std::array<SideLine, 4> lines{};
  for (int side = 0; side < 4; ++side) {
    const std::vector<cv::Point2f>& pts = samples[static_cast<size_t>(side)];
    const cv::Point2f direction = axis[static_cast<size_t>(side)];
    if (pts.size() < 12 || static_cast<float>(cv::norm(direction)) < 0.5f) {
      continue;
    }
    float lo = 1e9f;
    float hi = -1e9f;
    for (const cv::Point2f& point : pts) {
      const float along = point.dot(direction);
      lo = std::min(lo, along);
      hi = std::max(hi, along);
    }
    if (hi - lo < 0.35f * pitch) {
      continue;
    }
    cv::Vec4f fitted;
    cv::fitLine(pts, fitted, cv::DIST_HUBER, 0, 0.01, 0.01);
    SideLine line;
    line.direction = {fitted[0], fitted[1]};
    line.origin = {fitted[2], fitted[3]};
    if (static_cast<float>(cv::norm(line.direction)) < 1e-6f || !finite_point(line.origin) ||
        !finite_point(line.direction)) {
      continue;
    }
    if (line.direction.dot(direction) < 0.f) {
      line.direction *= -1.f;
    }
    line.thickness = side_thickness[static_cast<size_t>(side)];
    // Far enough from this edge's direction to be another contour, including at
    // a steep tilt where the image of the side is no longer parallel to the
    // lamp-affine edge. About twenty degrees.
    if (line.direction.dot(direction) < 0.940f) {
      continue;
    }
    line.ok = true;
    lines[static_cast<size_t>(side)] = line;
  }

  auto convex = [](const std::array<cv::Point2f, 4>& corners) {
    for (int i = 0; i < 4; ++i) {
      if (!finite_point(corners[static_cast<size_t>(i)])) {
        return false;
      }
      if (cross2(corners[static_cast<size_t>(i)], corners[static_cast<size_t>((i + 1) % 4)],
                 corners[static_cast<size_t>((i + 2) % 4)]) <= 0.f) {
        return false;
      }
    }
    return true;
  };

  int ready = 0;
  for (const SideLine& line : lines) {
    if (line.ok) {
      ++ready;
    }
  }
  // Top meets left, top meets right, bottom meets right, bottom meets left.
  const int meet[4][2] = {{0, 3}, {0, 1}, {2, 1}, {2, 3}};
  if (ready == 4) {
    std::array<cv::Point2f, 4> corners{};
    bool hits = true;
    for (int corner = 0; corner < 4; ++corner) {
      const SideLine& first = lines[static_cast<size_t>(meet[corner][0])];
      const SideLine& second = lines[static_cast<size_t>(meet[corner][1])];
      const float cross = first.direction.x * second.direction.y - first.direction.y * second.direction.x;
      if (std::abs(cross) < 1e-4f) {
        hits = false;
        break;
      }
      const cv::Point2f delta = second.origin - first.origin;
      const float t = (delta.x * second.direction.y - delta.y * second.direction.x) / cross;
      const cv::Point2f hit = first.origin + first.direction * t;
      if (!finite_point(hit) ||
          static_cast<float>(cv::norm(hit - quad[static_cast<size_t>(corner)])) > 0.25f * pitch) {
        hits = false;
        break;
      }
      corners[static_cast<size_t>(corner)] = hit;
    }
    if (hits && convex(corners)) {
      const std::vector<cv::Point2f> src = {
          {kPlateLo, kPlateLo},
          {kPlateHi, kPlateLo},
          {kPlateHi, kPlateHi},
          {kPlateLo, kPlateHi},
      };
      const std::vector<cv::Point2f> dst(corners.begin(), corners.end());
      const cv::Matx33d pose = from_mat(cv::getPerspectiveTransform(src, dst), 3);
      const float pose_pitch = imaged_pitch(pose);
      const float linear = static_cast<float>(pose(0, 0) * pose(1, 1) - pose(0, 1) * pose(1, 0));
      if (finite_h(pose) && linear > 0.f && pose_pitch > 0.75f * pitch && pose_pitch < 1.25f * pitch &&
          explains_observed(pose, corners, ells)) {
        from_lines = true;
        return corners;
      }
    }
  }
  return quad;
}

void consider(const cv::Mat& hsv, const cv::Matx33d& h, int support, int v_cut, int s_cut, std::vector<Fit>& passed,
              const std::vector<Blob>& ells, const std::vector<Blob>& pieces, const std::vector<Blob>& rims,
              bool tracked = false) {
  if (!finite_h(h) || std::abs(h(2, 2)) < 1e-8) {
    return;
  }
  std::array<cv::Point2f, 4> quad{};
  if (!plate_quad(h, quad) || !explains_observed(h, quad, ells)) {
    return;
  }
  const Score score = score_fit(hsv, h, support, v_cut, s_cut);
  // A carried plate already has its lamps. A mouth that is leaving the frame must
  // not throw that plate away when the fresh labeling has jumped.
  if (!score.pass && !(tracked && score.recall >= 0.45f && score.coverage >= 0.15f)) {
    return;
  }
  Fit fit;
  fit.detection.found = true;
  bool from_lines = false;
  const std::array<cv::Point2f, 4> measured =
      refine_plate(h, quad, ells, pieces, rims, hsv.cols, hsv.rows, from_lines);
  // The kept corners are this frame's pose: the four-edge homography when it
  // locks, otherwise the lamp affine or similarity with no corner edited alone.
  fit.detection.corners = measured;
  fit.measured = measured;
  fit.from_lines = from_lines;
  fit.detection.support = from_lines ? 4 : support;
  fit.detection.score = score.recall;
  fit.mouth_ok = score.mouth_ok;
  fit.mouth_in = score.mouth_in;
  fit.tracked = tracked;
  passed.push_back(fit);
}

bool better_fit(const Fit& a, const Fit& b) {
  const bool a_mouth = a.mouth_in >= 24;
  const bool b_mouth = b.mouth_in >= 24;
  if (a_mouth && b_mouth) {
    if (a.mouth_ok > b.mouth_ok + 0.20f) {
      return true;
    }
    if (b.mouth_ok > a.mouth_ok + 0.20f) {
      return false;
    }
  }
  if (a.detection.score > b.detection.score + 0.05f) {
    return true;
  }
  if (b.detection.score > a.detection.score + 0.05f) {
    return false;
  }
  if (a.detection.support != b.detection.support) {
    return a.detection.support > b.detection.support;
  }
  return a.detection.score > b.detection.score;
}

// Previous lamp-model plate. The drawn corners are the edge intersections of
// whichever labeling was kept. A miss clears this. Nothing is drawn from it
// unless the current frame still has lamps at the predicted places.
struct PlateMemory {
  bool found = false;
  std::array<cv::Point2f, 4> corners{};
  cv::Point2f velocity{};
};
PlateMemory& plate_memory() {
  static PlateMemory memory;
  return memory;
}

float median4(float a, float b, float c, float d) {
  float values[4] = {a, b, c, d};
  std::sort(values, values + 4);
  return 0.5f * (values[1] + values[2]);
}

bool quad_convex(const std::array<cv::Point2f, 4>& corners) {
  for (int i = 0; i < 4; ++i) {
    if (!finite_point(corners[static_cast<size_t>(i)])) {
      return false;
    }
    if (cross2(corners[static_cast<size_t>(i)], corners[static_cast<size_t>((i + 1) % 4)],
               corners[static_cast<size_t>((i + 2) % 4)]) <= 0.f) {
      return false;
    }
  }
  return true;
}

// Time identifies the lamps. The pose is whatever those lamps measure now.
// An empty frame clears the memory, so a miss is never filled from the frame before it.
Detection choose_fit(const std::vector<Fit>& passed) {
  PlateMemory& memory = plate_memory();
  if (passed.empty()) {
    memory.found = false;
    memory.velocity = {};
    return {};
  }
  const Fit* tracked = nullptr;
  const Fit* leader = nullptr;
  for (const Fit& fit : passed) {
    if (fit.tracked) {
      tracked = &fit;
      continue;
    }
    if (leader == nullptr || better_fit(fit, *leader)) {
      leader = &fit;
    }
  }
  const Fit* fresh = leader;
  if (fresh != nullptr && memory.found) {
    const float side = std::max(1.f, static_cast<float>(cv::norm(memory.corners[1] - memory.corners[0])));
    float nearest_dist = 0.35f * side;
    const Fit* nearest = nullptr;
    for (const Fit& fit : passed) {
      if (fit.tracked) {
        continue;
      }
      if (fit.detection.score + 0.05f < leader->detection.score) {
        continue;
      }
      if (leader->mouth_in >= 24 && fit.mouth_in >= 24 && fit.mouth_ok + 0.20f < leader->mouth_ok) {
        continue;
      }
      float dist = 0.f;
      for (int k = 0; k < 4; ++k) {
        dist = std::max(dist, static_cast<float>(cv::norm(fit.detection.corners[static_cast<size_t>(k)] -
                                                          memory.corners[static_cast<size_t>(k)])));
      }
      if (dist < nearest_dist) {
        nearest_dist = dist;
        nearest = &fit;
      }
    }
    if (nearest != nullptr) {
      fresh = nearest;
    }
  }
  // Four edges that meet are a measured homography and outrank a lamp pose that
  // could not lock them. Among those, the lamps named from the previous frame
  // win. Nothing here keeps the previous quad as a shape.
  const Fit* lined = nullptr;
  for (const Fit& fit : passed) {
    if (!fit.from_lines) {
      continue;
    }
    if (lined == nullptr || (fit.tracked && !lined->tracked) ||
        (fit.tracked == lined->tracked && better_fit(fit, *lined))) {
      lined = &fit;
    }
  }
  const Fit* chosen = lined != nullptr ? lined : (tracked != nullptr ? tracked : fresh);
  if (chosen == nullptr) {
    memory.found = false;
    memory.velocity = {};
    return {};
  }
  cv::Point2f velocity{};
  if (memory.found) {
    float dx[4];
    float dy[4];
    for (int k = 0; k < 4; ++k) {
      dx[k] = chosen->detection.corners[static_cast<size_t>(k)].x - memory.corners[static_cast<size_t>(k)].x;
      dy[k] = chosen->detection.corners[static_cast<size_t>(k)].y - memory.corners[static_cast<size_t>(k)].y;
    }
    velocity = {median4(dx[0], dx[1], dx[2], dx[3]), median4(dy[0], dy[1], dy[2], dy[3])};
  }
  memory.found = true;
  memory.corners = chosen->measured;
  memory.velocity = velocity;
  Detection result = chosen->detection;
  result.corners = chosen->measured;
  return result;
}

cv::Matx33d similarity(cv::Point2f m0, cv::Point2f m1, cv::Point2f i0, cv::Point2f i1) {
  const cv::Point2f vm = m1 - m0;
  const cv::Point2f vi = i1 - i0;
  const float vm_n = static_cast<float>(cv::norm(vm));
  const float vi_n = static_cast<float>(cv::norm(vi));
  cv::Matx33d h = cv::Matx33d::eye();
  if (vm_n < 1e-4f || vi_n < 1e-4f) {
    h(2, 2) = 0;
    return h;
  }
  const float scale = vi_n / vm_n;
  const float ang = std::atan2(vi.y, vi.x) - std::atan2(vm.y, vm.x);
  const float c = std::cos(ang) * scale;
  const float s = std::sin(ang) * scale;
  const cv::Point2f t = i0 - cv::Point2f(c * m0.x - s * m0.y, s * m0.x + c * m0.y);
  h(0, 0) = c;
  h(0, 1) = -s;
  h(0, 2) = t.x;
  h(1, 0) = s;
  h(1, 1) = c;
  h(1, 2) = t.y;
  return h;
}

cv::Point2f unit_vec(cv::Point2f v) {
  const float n = static_cast<float>(cv::norm(v));
  if (n < 1e-4f) {
    return {0.f, 0.f};
  }
  return v * (1.f / n);
}

bool positive_det(const cv::Matx33d& h) { return h(0, 0) * h(1, 1) - h(0, 1) * h(1, 0) > 0.0; }

bool corners_stay(const cv::Matx33d& base, const cv::Matx33d& refined, float pitch) {
  const cv::Point2f model[4] = {
      {kPlateLo, kPlateLo},
      {kPlateHi, kPlateLo},
      {kPlateHi, kPlateHi},
      {kPlateLo, kPlateHi},
  };
  for (const cv::Point2f& point : model) {
    if (cv::norm(apply(refined, point) - apply(base, point)) > 0.10f * pitch) {
      return false;
    }
  }
  return true;
}

struct PieceHit {
  const Blob* blob = nullptr;
  cv::Point2f model;
};

// Prefer the corner fragment. An end square is used only when the corner is absent.
// The area is compared with the area the drawing predicts at this frame's pitch.
PieceHit match_piece(const cv::Matx33d& h, float pitch, const std::vector<Blob>& pieces) {
  PieceHit hit;
  if (!(pitch > 1.f)) {
    return hit;
  }
  const float limit = 0.12f * pitch;
  const float px = pitch / kPitch;
  const std::array<PieceModel, 3>& models = piece_models();
  auto search = [&](int begin, int end) {
    const Blob* best = nullptr;
    cv::Point2f model;
    float best_dist = limit;
    for (int i = begin; i < end; ++i) {
      const cv::Point2f predicted = apply(h, models[static_cast<size_t>(i)].at);
      const float expected = models[static_cast<size_t>(i)].area * px * px;
      if (!(expected > 1.f)) {
        continue;
      }
      for (const Blob& blob : pieces) {
        const float dist = static_cast<float>(cv::norm(blob.centroid - predicted));
        if (dist > best_dist) {
          continue;
        }
        const float ratio = blob.area / expected;
        if (ratio < 0.75f || ratio > 1.40f) {
          continue;
        }
        best_dist = dist;
        best = &blob;
        model = models[static_cast<size_t>(i)].at;
      }
    }
    return PieceHit{best, model};
  };
  const PieceHit corner = search(0, 1);
  if (corner.blob != nullptr) {
    return corner;
  }
  return search(1, static_cast<int>(models.size()));
}

// A solid L seen at an angle is an affine image of the drawing, plus a small
// perspective residual across one lamp. Whitening by the covariance removes
// translation, rotation, scale and shear. What remains is a rotation, scored
// by the overlap of the filled canonical silhouettes.
struct Cloud {
  std::vector<cv::Point2d> pts;
  cv::Point2d mean;
  cv::Matx22d white = cv::Matx22d::eye();
  cv::Matx22d white_inv = cv::Matx22d::eye();
  bool ok = false;
};

Cloud whiten(const std::vector<cv::Point2d>& pts) {
  Cloud cloud;
  cloud.pts = pts;
  if (pts.size() < 8) {
    return cloud;
  }
  for (const cv::Point2d& point : pts) {
    cloud.mean += point;
  }
  cloud.mean *= 1.0 / static_cast<double>(pts.size());
  cv::Matx22d cov = cv::Matx22d::zeros();
  for (const cv::Point2d& point : pts) {
    const cv::Point2d d = point - cloud.mean;
    cov(0, 0) += d.x * d.x;
    cov(0, 1) += d.x * d.y;
    cov(1, 0) += d.y * d.x;
    cov(1, 1) += d.y * d.y;
  }
  cov *= 1.0 / static_cast<double>(pts.size());
  cv::Mat values;
  cv::Mat vectors;
  cv::SVD::compute(cv::Mat(cov), values, vectors, cv::noArray());
  if (values.rows < 2 || values.at<double>(1) < 1e-6 || values.at<double>(1) < 1e-3 * values.at<double>(0)) {
    return cloud;
  }
  cv::Matx22d basis;
  basis(0, 0) = vectors.at<double>(0, 0);
  basis(1, 0) = vectors.at<double>(1, 0);
  basis(0, 1) = vectors.at<double>(0, 1);
  basis(1, 1) = vectors.at<double>(1, 1);
  if (cv::determinant(basis) < 0.0) {
    basis(0, 0) = -basis(0, 0);
    basis(1, 0) = -basis(1, 0);
  }
  const double s0 = 1.0 / std::sqrt(values.at<double>(0));
  const double s1 = 1.0 / std::sqrt(values.at<double>(1));
  const cv::Matx22d root(s0, 0.0, 0.0, s1);
  const cv::Matx22d root_inv(1.0 / s0, 0.0, 0.0, 1.0 / s1);
  cloud.white = root * basis.t();
  cloud.white_inv = basis * root_inv;
  cloud.ok = true;
  return cloud;
}

std::vector<cv::Point2d> sample_poly(const cv::Point2d* poly, int n, double step) {
  std::vector<cv::Point2d> out;
  for (int i = 0; i < n; ++i) {
    const cv::Point2d a = poly[i];
    const cv::Point2d b = poly[(i + 1) % n];
    const double len = cv::norm(b - a);
    const int pieces = std::max(1, static_cast<int>(std::ceil(len / step)));
    for (int k = 0; k < pieces; ++k) {
      out.push_back(a + (b - a) * (static_cast<double>(k) / static_cast<double>(pieces)));
    }
  }
  return out;
}

const Cloud& solid_cloud() {
  static const Cloud cloud = [] {
    const cv::Point2d poly[] = {{0, 0}, {30, 0}, {30, 8}, {8, 8}, {8, 30}, {0, 30}};
    return whiten(sample_poly(poly, 6, 0.4));
  }();
  return cloud;
}

constexpr int kCanon = 96;

cv::Mat canon_mask(const std::vector<cv::Point2d>& whitened) {
  cv::Mat mask(kCanon, kCanon, CV_8UC1, cv::Scalar(0));
  std::vector<cv::Point> pix;
  pix.reserve(whitened.size());
  constexpr double kScale = 14.0;
  for (const cv::Point2d& point : whitened) {
    pix.emplace_back(cvRound(kCanon * 0.5 + point.x * kScale), cvRound(kCanon * 0.5 + point.y * kScale));
  }
  const std::vector<std::vector<cv::Point>> polys = {pix};
  cv::fillPoly(mask, polys, cv::Scalar(255));
  return mask;
}

// 96*96 bits, one word per 64 pixels. Overlap is a population count, so each
// contour pays for one raster of itself and then reuses the solid-L masks.
constexpr int kCanonWords = (kCanon * kCanon) / 64;
using BitMask = std::array<std::uint64_t, kCanonWords>;

BitMask pack_mask(const cv::Mat& mask) {
  BitMask bits{};
  int index = 0;
  for (int row = 0; row < mask.rows; ++row) {
    const uchar* pix = mask.ptr<uchar>(row);
    for (int col = 0; col < mask.cols; ++col, ++index) {
      if (pix[col] != 0) {
        bits[static_cast<size_t>(index >> 6)] |= std::uint64_t{1} << (index & 63);
      }
    }
  }
  return bits;
}

double bit_iou(const BitMask& a, const BitMask& b) {
  int both = 0;
  int either = 0;
  for (int i = 0; i < kCanonWords; ++i) {
    const std::uint64_t aw = a[static_cast<size_t>(i)];
    const std::uint64_t bw = b[static_cast<size_t>(i)];
    both += __builtin_popcountll(aw & bw);
    either += __builtin_popcountll(aw | bw);
  }
  return either < 1 ? 0.0 : static_cast<double>(both) / static_cast<double>(either);
}

const BitMask& model_bits_deg(int deg) {
  static const std::array<BitMask, 180> masks = [] {
    std::array<BitMask, 180> built{};
    const Cloud& model = solid_cloud();
    std::vector<cv::Point2d> model_w;
    model_w.reserve(model.pts.size());
    for (const cv::Point2d& point : model.pts) {
      model_w.push_back(model.white * (point - model.mean));
    }
    for (int step = 0; step < 180; ++step) {
      const double rad = (step * 2) * CV_PI / 180.0;
      const double c = std::cos(rad);
      const double s = std::sin(rad);
      std::vector<cv::Point2d> spun;
      spun.reserve(model_w.size());
      for (const cv::Point2d& point : model_w) {
        spun.emplace_back(c * point.x - s * point.y, s * point.x + c * point.y);
      }
      built[static_cast<size_t>(step)] = pack_mask(canon_mask(spun));
    }
    return built;
  }();
  int wrapped = deg % 360;
  if (wrapped < 0) {
    wrapped += 360;
  }
  return masks[static_cast<size_t>(wrapped / 2)];
}

struct ShapeHit {
  float score = 0.f;
  double angle = 0.0;
  bool ok = false;
};

ShapeHit match_solid(const Cloud& image) {
  ShapeHit hit;
  const Cloud& model = solid_cloud();
  if (!model.ok || !image.ok) {
    return hit;
  }
  std::vector<cv::Point2d> image_w;
  image_w.reserve(image.pts.size());
  for (const cv::Point2d& point : image.pts) {
    image_w.push_back(image.white * (point - image.mean));
  }
  const BitMask image_bits = pack_mask(canon_mask(image_w));
  auto score_at = [&](int deg) { return bit_iou(model_bits_deg(deg), image_bits); };
  int best_deg = 0;
  double best = -1.0;
  for (int deg = 0; deg < 360; deg += 8) {
    const double score = score_at(deg);
    if (score > best) {
      best = score;
      best_deg = deg;
    }
  }
  for (int deg = best_deg - 8; deg <= best_deg + 8; deg += 2) {
    int wrapped = deg % 360;
    if (wrapped < 0) {
      wrapped += 360;
    }
    const double score = score_at(wrapped);
    if (score > best) {
      best = score;
      best_deg = wrapped;
    }
  }
  hit.score = static_cast<float>(best);
  hit.angle = best_deg * CV_PI / 180.0;
  hit.ok = true;
  return hit;
}

cv::Point2f semantic_image(const Cloud& image, double angle, cv::Point2d model_pt) {
  const Cloud& model = solid_cloud();
  const cv::Point2d y = model.white * (model_pt - model.mean);
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const cv::Point2d z(c * y.x - s * y.y, s * y.x + c * y.y);
  const cv::Point2d out = image.mean + image.white_inv * z;
  return {static_cast<float>(out.x), static_cast<float>(out.y)};
}

Blob make_blob(const std::vector<cv::Point>& contour) {
  Blob blob;
  const double area = std::abs(cv::contourArea(contour));
  if (area < 1.0 || contour.size() < 8) {
    return blob;
  }
  const cv::Moments moments = cv::moments(contour);
  if (moments.m00 == 0) {
    return blob;
  }
  blob.centroid = cv::Point2f(static_cast<float>(moments.m10 / moments.m00), static_cast<float>(moments.m01 / moments.m00));
  blob.area = static_cast<float>(area);
  std::vector<cv::Point> hull;
  cv::convexHull(contour, hull);
  const double hull_area = std::abs(cv::contourArea(hull));
  blob.solidity = hull_area > 1.0 ? static_cast<float>(area / hull_area) : 0.f;
  const cv::RotatedRect rect = cv::minAreaRect(contour);
  const float rw = std::max(rect.size.width, 1.f);
  const float rh = std::max(rect.size.height, 1.f);
  blob.aspect = std::min(rw, rh) / std::max(rw, rh);
  blob.extent = blob.area / (rw * rh);
  blob.mean_side = 0.5f * (rw + rh);
  blob.elbow = blob.centroid - rect.center;
  cv::Point2f rect_pts[4];
  rect.points(rect_pts);
  blob.arm_x = unit_vec(rect_pts[1] - rect_pts[0]);
  blob.arm_y = unit_vec(rect_pts[2] - rect_pts[1]);
  std::vector<int> hull_idx;
  cv::convexHull(contour, hull_idx, false, false);
  if (hull_idx.size() >= 3) {
    try {
      std::vector<cv::Vec4i> defects;
      cv::convexityDefects(contour, hull_idx, defects);
      float depth = 0.f;
      for (const cv::Vec4i& defect : defects) {
        depth = std::max(depth, defect[3] / 256.f);
      }
      blob.defect = depth / std::sqrt(rw * rh);
    } catch (const cv::Exception&) {
      blob.defect = 0.f;
    }
  }
  const float elbow_ratio = static_cast<float>(cv::norm(blob.elbow)) / blob.mean_side;
  // Frame 0 lamps overlap the canonical L at 0.88–0.92. The dimmer full lamp
  // on frame 459 is 0.736. Bars and empty frames stay at or below 0.67.
  std::vector<cv::Point2d> boundary;
  boundary.reserve(contour.size());
  for (const cv::Point& point : contour) {
    boundary.emplace_back(point.x, point.y);
  }
  const bool worth_shape = contour.size() >= 24;
  const Cloud image = worth_shape ? whiten(boundary) : Cloud{};
  const ShapeHit shape = worth_shape ? match_solid(image) : ShapeHit{};
  constexpr float kShape = 0.72f;
  blob.is_l = shape.ok && shape.score >= kShape;
  auto keep_edge = [&]() {
    blob.edge.reserve(contour.size());
    for (const cv::Point& point : contour) {
      blob.edge.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }
  };
  if (blob.is_l) {
    const cv::Point2d semantic[4] = {{0, 0}, {8, 8}, {30, 0}, {0, 30}};
    for (int i = 0; i < 4; ++i) {
      blob.semantic[i] = semantic_image(image, shape.angle, semantic[i]);
    }
    blob.quality = shape.score;
    keep_edge();
    return blob;
  }
  const bool piece = blob.solidity >= 0.80f && blob.extent >= 0.62f && blob.aspect >= 0.55f && blob.defect <= 0.42f;
  if (!piece) {
    if (contour.size() < 24) {
      blob.area = 0.f;
    } else {
      keep_edge();
    }
    return blob;
  }
  blob.is_piece = true;
  blob.quality = -(std::abs(blob.solidity - 0.632f) + std::abs(blob.extent - 0.462f) + std::abs(blob.defect - 0.47f) +
                   std::abs(elbow_ratio - 0.219f));
  keep_edge();
  return blob;
}

struct Parts {
  std::vector<Blob> ells;
  std::vector<Blob> pieces;
  // Bright contours that are neither a solid L nor a corner fragment. Line fitting
  // can use a clipped arm; these blobs do not open a pose.
  std::vector<Blob> rims;
  int v_cut = 0;
  int s_cut = 0;
};

Parts extract_parts(const cv::Mat& hsv) {
  Parts parts;
  std::vector<int> v_hist(256, 0);
  for (int row = 0; row < hsv.rows; ++row) {
    const cv::Vec3b* pix = hsv.ptr<cv::Vec3b>(row);
    for (int col = 0; col < hsv.cols; ++col) {
      ++v_hist[pix[col][2]];
    }
  }
  parts.v_cut = triangle_threshold(v_hist, 0, 255);
  std::vector<int> s_hist(256, 0);
  for (int row = 0; row < hsv.rows; ++row) {
    const cv::Vec3b* pix = hsv.ptr<cv::Vec3b>(row);
    for (int col = 0; col < hsv.cols; ++col) {
      if (pix[col][2] >= parts.v_cut) {
        ++s_hist[pix[col][1]];
      }
    }
  }
  // A one-bin spike at saturation 0 can outvote the broader lamp peak.
  // The mode is taken on a short smooth; the foot stays on the raw histogram.
  std::vector<float> smooth(256, 0.f);
  constexpr int kSmooth = 8;
  for (int i = 0; i < 256; ++i) {
    float sum = 0.f;
    int count = 0;
    for (int k = -kSmooth; k <= kSmooth; ++k) {
      const int bin = i + k;
      if (bin >= 0 && bin < 256) {
        sum += static_cast<float>(s_hist[static_cast<size_t>(bin)]);
        ++count;
      }
    }
    smooth[static_cast<size_t>(i)] = sum / static_cast<float>(count);
  }
  int smooth_mode = 0;
  for (int i = 1; i < 256; ++i) {
    if (smooth[static_cast<size_t>(i)] > smooth[static_cast<size_t>(smooth_mode)]) {
      smooth_mode = i;
    }
  }
  parts.s_cut = triangle_threshold(s_hist, smooth_mode, 255);

  cv::Mat mask;
  cv::inRange(hsv, cv::Scalar(0, 0, parts.v_cut), cv::Scalar(179, parts.s_cut, 255), mask);
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  std::vector<Blob> blobs;
  for (const std::vector<cv::Point>& contour : contours) {
    Blob blob = make_blob(contour);
    if (blob.area > 0.f) {
      blobs.push_back(blob);
    }
  }
  std::sort(blobs.begin(), blobs.end(), [](const Blob& a, const Blob& b) {
    if (a.is_l != b.is_l) {
      return a.is_l > b.is_l;
    }
    return a.quality > b.quality;
  });
  std::vector<Blob> kept;
  for (const Blob& blob : blobs) {
    bool duplicate = false;
    for (const Blob& prior : kept) {
      if (prior.is_l != blob.is_l || prior.is_piece != blob.is_piece) {
        continue;
      }
      const float limit = 0.45f * std::max(prior.mean_side, blob.mean_side);
      if (cv::norm(prior.centroid - blob.centroid) < limit) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      kept.push_back(blob);
    }
  }
  const float x_hi = static_cast<float>(hsv.cols - 1);
  const float y_hi = static_cast<float>(hsv.rows - 1);
  auto cut_by_frame = [&](const Blob& blob) {
    for (const cv::Point2f& point : blob.edge) {
      if (point.x <= 0.f || point.y <= 0.f || point.x >= x_hi || point.y >= y_hi) {
        return true;
      }
    }
    return false;
  };
  for (const Blob& blob : kept) {
    // A contour that meets the image border is not a whole lamp. Its centroid
    // is not the model centroid, so it can only contribute edge points.
    if (blob.is_l && !cut_by_frame(blob)) {
      parts.ells.push_back(blob);
    } else if (blob.is_piece && !cut_by_frame(blob)) {
      parts.pieces.push_back(blob);
    } else {
      parts.rims.push_back(blob);
    }
  }
  if (parts.ells.size() > 8) {
    parts.ells.resize(8);
  }
  if (parts.rims.size() > 12) {
    std::sort(parts.rims.begin(), parts.rims.end(), [](const Blob& a, const Blob& b) { return a.area > b.area; });
    parts.rims.resize(12);
  }
  return parts;
}

// The gapped corner is one correspondence of its own once the solid lamps are gone.
// The corner notch measures near 0.21 after bloom; the drawing notch is 0.303 and an
// end square stays near 0.09, so the two populations do not overlap. The corner is
// 160/64 = 2.5 times an end, and the recovered pitch has to predict those areas.
void consider_gapped(const cv::Mat& hsv, const Parts& parts, std::vector<Fit>& passed) {
  const std::vector<Blob>& pieces = parts.pieces;
  const std::array<PieceModel, 3>& models = piece_models();
  const int count = static_cast<int>(pieces.size());
  for (int i = 0; i < count; ++i) {
    const Blob& corner = pieces[static_cast<size_t>(i)];
    if (corner.defect < 0.16f || corner.defect > 0.35f) {
      continue;
    }
    for (int j = 0; j < count; ++j) {
      if (j == i) {
        continue;
      }
      const Blob& end = pieces[static_cast<size_t>(j)];
      if (end.defect > 0.18f) {
        continue;
      }
      const float ratio = corner.area / end.area;
      if (ratio < 2.1f || ratio > 3.2f) {
        continue;
      }
      for (int end_id = 1; end_id <= 2; ++end_id) {
        const PieceModel& end_model = models[static_cast<size_t>(end_id)];
        const cv::Matx33d sim = similarity(models[0].at, end_model.at, corner.centroid, end.centroid);
        if (!elbow_agrees(sim, models[0].at, models[0].elbow, corner.elbow)) {
          continue;
        }
        const float pitch = imaged_pitch(sim);
        if (!(pitch > 1.f)) {
          continue;
        }
        const float scale = pitch / kPitch;
        const float corner_fit = corner.area / (models[0].area * scale * scale);
        const float end_fit = end.area / (end_model.area * scale * scale);
        if (corner_fit < 0.75f || corner_fit > 1.40f || end_fit < 0.75f || end_fit > 1.40f) {
          continue;
        }
        consider(hsv, sim, 2, parts.v_cut, parts.s_cut, passed, parts.ells, parts.pieces, parts.rims);
        const int other_id = 3 - end_id;
        const PieceModel& other_model = models[static_cast<size_t>(other_id)];
        const cv::Point2f predicted = apply(sim, other_model.at);
        const float expected = other_model.area * scale * scale;
        const Blob* third = nullptr;
        float nearest = 0.12f * pitch;
        for (int k = 0; k < count; ++k) {
          if (k == i || k == j) {
            continue;
          }
          const Blob& blob = pieces[static_cast<size_t>(k)];
          if (blob.defect > 0.18f) {
            continue;
          }
          const float dist = static_cast<float>(cv::norm(blob.centroid - predicted));
          if (dist >= nearest) {
            continue;
          }
          const float fit = blob.area / expected;
          if (fit < 0.75f || fit > 1.40f) {
            continue;
          }
          nearest = dist;
          third = &blob;
        }
        if (third == nullptr) {
          continue;
        }
        cv::Point2f src[3] = {models[0].at, end_model.at, other_model.at};
        cv::Point2f dst[3] = {corner.centroid, end.centroid, third->centroid};
        const cv::Mat affine_mat = cv::getAffineTransform(src, dst);
        if (affine_mat.empty()) {
          continue;
        }
        const cv::Matx33d affine = from_mat(affine_mat, 2);
        if (!positive_det(affine) || !corners_stay(sim, affine, pitch) ||
            !elbow_agrees(affine, models[0].at, models[0].elbow, corner.elbow)) {
          continue;
        }
        consider(hsv, affine, 3, parts.v_cut, parts.s_cut, passed, parts.ells, parts.pieces, parts.rims);
      }
    }
  }
}

struct SolidGeom {
  cv::Point2f at[4];
};

const std::array<SolidGeom, 3>& solid_geom() {
  static const std::array<SolidGeom, 3> geom = [] {
    const cv::Point2f draw[3][4] = {
        {{0.f, 0.f}, {8.f, 8.f}, {30.f, 0.f}, {0.f, 30.f}},
        {{0.f, 80.f}, {8.f, 72.f}, {0.f, 50.f}, {30.f, 80.f}},
        {{80.f, 80.f}, {72.f, 72.f}, {50.f, 80.f}, {80.f, 50.f}},
    };
    std::array<SolidGeom, 3> built{};
    for (int lamp = 0; lamp < 3; ++lamp) {
      for (int k = 0; k < 4; ++k) {
        built[static_cast<size_t>(lamp)].at[k] = to_model(draw[lamp][k].x, draw[lamp][k].y);
      }
    }
    return built;
  }();
  return geom;
}

cv::Matx33d affine_fit(const std::vector<cv::Point2f>& src, const std::vector<cv::Point2f>& dst) {
  cv::Matx33d h = cv::Matx33d::eye();
  h(2, 2) = 0;
  const int n = static_cast<int>(src.size());
  if (n < 3 || dst.size() != src.size()) {
    return h;
  }
  cv::Mat design(2 * n, 6, CV_64F, cv::Scalar(0));
  cv::Mat image(2 * n, 1, CV_64F, cv::Scalar(0));
  for (int i = 0; i < n; ++i) {
    const double x = src[static_cast<size_t>(i)].x;
    const double y = src[static_cast<size_t>(i)].y;
    design.at<double>(2 * i, 0) = x;
    design.at<double>(2 * i, 1) = y;
    design.at<double>(2 * i, 2) = 1.0;
    design.at<double>(2 * i + 1, 3) = x;
    design.at<double>(2 * i + 1, 4) = y;
    design.at<double>(2 * i + 1, 5) = 1.0;
    image.at<double>(2 * i, 0) = dst[static_cast<size_t>(i)].x;
    image.at<double>(2 * i + 1, 0) = dst[static_cast<size_t>(i)].y;
  }
  cv::Mat solved;
  if (!cv::solve(design, image, solved, cv::DECOMP_SVD)) {
    return h;
  }
  h = cv::Matx33d::eye();
  h(0, 0) = solved.at<double>(0);
  h(0, 1) = solved.at<double>(1);
  h(0, 2) = solved.at<double>(2);
  h(1, 0) = solved.at<double>(3);
  h(1, 1) = solved.at<double>(4);
  h(1, 2) = solved.at<double>(5);
  return h;
}

void append_lamp(int lamp, const Blob& blob, std::vector<cv::Point2f>& src, std::vector<cv::Point2f>& dst) {
  const SolidGeom& geom = solid_geom()[static_cast<size_t>(lamp)];
  for (int k = 0; k < 4; ++k) {
    src.push_back(geom.at[k]);
    dst.push_back(blob.semantic[k]);
  }
}

// Centroids fix the similarity. The whitened arms have to agree with it, and a third
// lamp has to land on that same similarity. An adjacent pair still has one 90°
// relabeling with the same arms; the gapped mouths are what reject that labeling.
bool arms_match(const cv::Matx33d& h, int lamp, const Blob& blob) {
  const SolidGeom& geom = solid_geom()[static_cast<size_t>(lamp)];
  for (int tip = 2; tip <= 3; ++tip) {
    const cv::Point2f predicted = apply(h, geom.at[tip]) - apply(h, geom.at[0]);
    const cv::Point2f observed = blob.semantic[tip] - blob.semantic[0];
    const float pn = static_cast<float>(cv::norm(predicted));
    const float on = static_cast<float>(cv::norm(observed));
    if (pn < 1e-3f || on < 1e-3f) {
      return false;
    }
    if (predicted.dot(observed) / (pn * on) < 0.50f) {
      return false;
    }
  }
  return true;
}

bool identity_ok(const std::vector<std::pair<int, const Blob*>>& used) {
  if (used.size() < 2) {
    return true;
  }
  const cv::Matx33d pose = similarity(kLamps[used[0].first].at, kLamps[used[1].first].at, used[0].second->centroid,
                                      used[1].second->centroid);
  if (!positive_det(pose)) {
    return false;
  }
  const float pitch = imaged_pitch(pose);
  if (!(pitch > 1.f)) {
    return false;
  }
  for (int i = 0; i < static_cast<int>(used.size()); ++i) {
    if (!arms_match(pose, used[static_cast<size_t>(i)].first, *used[static_cast<size_t>(i)].second)) {
      return false;
    }
    if (i >= 2 && cv::norm(apply(pose, kLamps[used[static_cast<size_t>(i)].first].at) -
                           used[static_cast<size_t>(i)].second->centroid) > 0.15f * pitch) {
      return false;
    }
  }
  return true;
}

// One lamp does not name the plate. Another solid L inside the plate has to be one of
// the remaining lamps under the same centroid similarity.
bool agrees_with_visible(const cv::Matx33d& pose, int lamp, const Blob& self, const std::vector<Blob>& ells) {
  std::array<cv::Point2f, 4> quad{};
  if (!plate_quad(pose, quad)) {
    return false;
  }
  const float pitch = imaged_pitch(pose);
  if (!(pitch > 1.f)) {
    return false;
  }
  const std::array<cv::Point2f, 4> region = grow_quad(quad, 1.20f);
  for (const Blob& other : ells) {
    if (&other == &self) {
      continue;
    }
    if (!point_in_quad(region, other.centroid)) {
      continue;
    }
    bool matched = false;
    for (int other_id = 0; other_id < 3; ++other_id) {
      if (other_id == lamp) {
        continue;
      }
      const std::vector<std::pair<int, const Blob*>> used = {{lamp, &self}, {other_id, &other}};
      if (!identity_ok(used)) {
        continue;
      }
      if (cv::norm(apply(pose, kLamps[other_id].at) - other.centroid) <= 0.20f * pitch) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
}

// Prediction only names the lamps. Two or three named lamps are fit again, so
// rotation and foreshortening come from this frame. One lamp cannot show a turn.
void continue_from_previous(const cv::Mat& hsv, const std::vector<Blob>& ells, const std::vector<Blob>& pieces,
                            const std::vector<Blob>& rims, int v_cut, int s_cut, std::vector<Fit>& passed) {
  const PlateMemory& memory = plate_memory();
  if (!memory.found || !quad_convex(memory.corners)) {
    return;
  }
  const std::vector<cv::Point2f> src = {{kPlateLo, kPlateLo}, {kPlateHi, kPlateLo}, {kPlateHi, kPlateHi}, {kPlateLo, kPlateHi}};
  const std::vector<cv::Point2f> dst(memory.corners.begin(), memory.corners.end());
  const cv::Matx33d previous = from_mat(cv::getPerspectiveTransform(src, dst), 3);
  if (!finite_h(previous)) {
    return;
  }
  const float pitch = imaged_pitch(previous);
  if (!(pitch > 1.f)) {
    return;
  }
  const Blob* matched[3] = {nullptr, nullptr, nullptr};
  std::vector<char> taken(ells.size(), 0);
  int count = 0;
  for (int lamp = 0; lamp < 3; ++lamp) {
    const cv::Point2f predicted = apply(previous, kLamps[lamp].at) + memory.velocity;
    int best = -1;
    float best_dist = 0.45f * pitch;
    for (int i = 0; i < static_cast<int>(ells.size()); ++i) {
      if (taken[static_cast<size_t>(i)] != 0) {
        continue;
      }
      const float dist = static_cast<float>(cv::norm(ells[static_cast<size_t>(i)].centroid - predicted));
      if (dist < best_dist) {
        best_dist = dist;
        best = i;
      }
    }
    if (best < 0) {
      continue;
    }
    taken[static_cast<size_t>(best)] = 1;
    matched[lamp] = &ells[static_cast<size_t>(best)];
    ++count;
  }
  if (count < 1) {
    return;
  }
  if (count == 1) {
    cv::Point2f shift(0.f, 0.f);
    for (int lamp = 0; lamp < 3; ++lamp) {
      if (matched[lamp] == nullptr) {
        continue;
      }
      shift = matched[lamp]->centroid - apply(previous, kLamps[lamp].at);
    }
    std::array<cv::Point2f, 4> moved{};
    for (int k = 0; k < 4; ++k) {
      moved[static_cast<size_t>(k)] = memory.corners[static_cast<size_t>(k)] + shift;
    }
    if (!quad_convex(moved)) {
      return;
    }
    const std::vector<cv::Point2f> moved_dst(moved.begin(), moved.end());
    const cv::Matx33d pose = from_mat(cv::getPerspectiveTransform(src, moved_dst), 3);
    consider(hsv, pose, 2, v_cut, s_cut, passed, ells, pieces, rims, true);
    return;
  }
  std::vector<cv::Point2f> model;
  std::vector<cv::Point2f> image;
  model.reserve(static_cast<size_t>(count));
  image.reserve(static_cast<size_t>(count));
  for (int lamp = 0; lamp < 3; ++lamp) {
    if (matched[lamp] == nullptr) {
      continue;
    }
    model.push_back(kLamps[lamp].at);
    image.push_back(matched[lamp]->centroid);
  }
  const cv::Matx33d pose = count >= 3 ? affine_fit(model, image) : similarity(model[0], model[1], image[0], image[1]);
  if (!positive_det(pose)) {
    return;
  }
  consider(hsv, pose, count >= 3 ? 3 : 2, v_cut, s_cut, passed, ells, pieces, rims, true);
}

}  // namespace

Detection detect_marker(const cv::Mat& bgr) {
  Detection empty;
  if (bgr.empty()) {
    return empty;
  }
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
  const Parts parts = extract_parts(hsv);
  const std::vector<Blob>& ells = parts.ells;
  const std::vector<Blob>& pieces = parts.pieces;
  std::vector<Fit> passed;
  const int n = static_cast<int>(ells.size());
  auto accept = [&](const cv::Matx33d& h, int support) {
    consider(hsv, h, support, parts.v_cut, parts.s_cut, passed, ells, pieces, parts.rims);
  };
  continue_from_previous(hsv, ells, pieces, parts.rims, parts.v_cut, parts.s_cut, passed);

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      for (int k = j + 1; k < n; ++k) {
        const Blob* trio[3] = {&ells[static_cast<size_t>(i)], &ells[static_cast<size_t>(j)], &ells[static_cast<size_t>(k)]};
        const int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        for (const auto& perm : perms) {
          const std::vector<std::pair<int, const Blob*>> used = {
              {0, trio[perm[0]]},
              {1, trio[perm[1]]},
              {2, trio[perm[2]]},
          };
          if (!identity_ok(used)) {
            continue;
          }
          std::vector<cv::Point2f> src;
          std::vector<cv::Point2f> dst;
          for (const auto& item : used) {
            src.push_back(kLamps[item.first].at);
            dst.push_back(item.second->centroid);
          }
          const cv::Matx33d affine = affine_fit(src, dst);
          if (positive_det(affine)) {
            accept(affine, 3);
          }
        }
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          if (a == b) {
            continue;
          }
          const std::vector<std::pair<int, const Blob*>> used = {
              {a, &ells[static_cast<size_t>(i)]},
              {b, &ells[static_cast<size_t>(j)]},
          };
          if (!identity_ok(used)) {
            continue;
          }
          const cv::Matx33d sim = similarity(kLamps[a].at, kLamps[b].at, ells[static_cast<size_t>(i)].centroid,
                                             ells[static_cast<size_t>(j)].centroid);
          if (positive_det(sim)) {
            accept(sim, 2);
          }
        }
      }
    }
  }

  if (passed.empty()) {
    for (const Blob& blob : ells) {
      for (int lamp = 0; lamp < 3; ++lamp) {
        std::vector<cv::Point2f> src;
        std::vector<cv::Point2f> dst;
        append_lamp(lamp, blob, src, dst);
        const cv::Matx33d affine = affine_fit(src, dst);
        if (!positive_det(affine) || !agrees_with_visible(affine, lamp, blob, ells)) {
          continue;
        }
        const PieceHit hit = match_piece(affine, imaged_pitch(affine), pieces);
        accept(affine, hit.blob != nullptr ? 2 : 1);
      }
    }
  }

  if (passed.empty()) {
    consider_gapped(hsv, parts, passed);
  }
  return choose_fit(passed);
}

void draw_detection(cv::Mat& bgr, const Detection& detection) {
  if (!detection.found) {
    cv::putText(bgr, "undetected", {24, 48}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 220, 255}, 2, cv::LINE_AA);
    return;
  }
  const cv::Scalar colors[4] = {{0, 255, 255}, {255, 255, 0}, {255, 0, 255}, {0, 165, 255}};
  const char* names[4] = {"LT", "RT", "RB", "LB"};
  constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
  constexpr double kScale = 0.6;
  constexpr int kThickness = 2;
  constexpr float kRing = 5.f;
  cv::Point2f center(0.f, 0.f);
  for (const cv::Point2f& corner : detection.corners) {
    center += corner;
  }
  center *= 0.25f;
  for (int i = 0; i < 4; ++i) {
    cv::Point a(cvRound(detection.corners[static_cast<size_t>(i)].x), cvRound(detection.corners[static_cast<size_t>(i)].y));
    cv::Point b(cvRound(detection.corners[static_cast<size_t>((i + 1) % 4)].x),
                cvRound(detection.corners[static_cast<size_t>((i + 1) % 4)].y));
    if (cv::clipLine(cv::Rect(0, 0, bgr.cols, bgr.rows), a, b)) {
      cv::line(bgr, a, b, {0, 220, 0}, 2, cv::LINE_AA);
    }
  }
  for (int i = 0; i < 4; ++i) {
    const cv::Point2f corner = detection.corners[static_cast<size_t>(i)];
    if (corner.x < 0.f || corner.y < 0.f || corner.x >= static_cast<float>(bgr.cols) ||
        corner.y >= static_cast<float>(bgr.rows)) {
      continue;
    }
    cv::circle(bgr, corner, cvRound(kRing), colors[i], 2, cv::LINE_AA);
    cv::Point2f outward = corner - center;
    const float span = static_cast<float>(cv::norm(outward));
    if (!(span > 1.f)) {
      continue;
    }
    outward *= 1.f / span;
    int baseline = 0;
    const cv::Size text = cv::getTextSize(names[i], kFont, kScale, kThickness, &baseline);
    baseline += kThickness;
    // The inner corner of each label sits on the outward diagonal, at the same
    // distance from its ring. The glyphs extend into the outer quadrant.
    const cv::Point2f anchor = corner + outward * (kRing + 7.f);
    const float origin_x = outward.x >= 0.f ? anchor.x : anchor.x - static_cast<float>(text.width);
    const float origin_y = outward.y >= 0.f ? anchor.y + static_cast<float>(text.height) : anchor.y - static_cast<float>(baseline);
    if (origin_x < 1.f || origin_y < static_cast<float>(text.height) || origin_x + static_cast<float>(text.width) >= static_cast<float>(bgr.cols) - 1 ||
        origin_y + static_cast<float>(baseline) >= static_cast<float>(bgr.rows) - 1) {
      continue;
    }
    cv::putText(bgr, names[i], cv::Point(cvRound(origin_x), cvRound(origin_y)), kFont, kScale, colors[i], kThickness, cv::LINE_AA);
  }
}
