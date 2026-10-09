#include "marker_model.hpp"

#include <opencv2/core/persistence.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace {

bool read_number(const cv::FileStorage& fs, const char* key, double& out, std::string& error) {
  const cv::FileNode node = fs[key];
  if (node.empty()) {
    error = std::string("missing ") + key;
    return false;
  }
  out = static_cast<double>(node);
  return true;
}

struct CornerAxis {
  cv::Point2d origin;
  cv::Point2d u;
  cv::Point2d v;
};

}  // namespace

bool MarkerModel::arm_lit(int corner, double distance) const {
  if (distance < 0 || distance >= arm) {
    return false;
  }
  if (corner != notch_corner) {
    return true;
  }
  if (distance < notch_bright) {
    return true;
  }
  if (distance < notch_bright + notch_gap) {
    return false;
  }
  return distance < arm;
}

bool MarkerModel::frame_lit(double x, double y) const {
  auto on_strip = [&](int edge, double c) {
    if (edge == 0) {
      if (c < arm) {
        return arm_lit(0, c);
      }
      if (c > outer - arm) {
        return arm_lit(1, outer - c);
      }
      return false;
    }
    if (edge == 1) {
      if (c < arm) {
        return arm_lit(1, c);
      }
      if (c > outer - arm) {
        return arm_lit(2, outer - c);
      }
      return false;
    }
    if (edge == 2) {
      if (c < arm) {
        return arm_lit(3, c);
      }
      if (c > outer - arm) {
        return arm_lit(2, outer - c);
      }
      return false;
    }
    if (c < arm) {
      return arm_lit(0, c);
    }
    if (c > outer - arm) {
      return arm_lit(3, outer - c);
    }
    return false;
  };

  bool lit = false;
  if (y >= 0 && y < frame) {
    lit = lit || on_strip(0, x);
  }
  if (y > outer - frame && y <= outer) {
    lit = lit || on_strip(2, x);
  }
  if (x > outer - frame && x <= outer) {
    lit = lit || on_strip(1, y);
  }
  if (x >= 0 && x < frame) {
    lit = lit || on_strip(3, y);
  }
  return lit;
}

std::string MarkerModel::edge_runs(int edge) const {
  std::string text;
  bool any = false;
  bool current = false;
  int count = 0;
  const int steps = static_cast<int>(std::lround(outer));
  for (int s = 0; s < steps; ++s) {
    const bool lit = [&] {
      const double c = s + 0.5;
      if (edge == 0) {
        return frame_lit(c, frame * 0.5);
      }
      if (edge == 1) {
        return frame_lit(outer - frame * 0.5, c);
      }
      if (edge == 2) {
        return frame_lit(c, outer - frame * 0.5);
      }
      return frame_lit(frame * 0.5, c);
    }();
    if (!any) {
      any = true;
      current = lit;
      count = 1;
    } else if (lit == current) {
      ++count;
    } else {
      text.push_back(current ? 'B' : 'D');
      text += std::to_string(count);
      current = lit;
      count = 1;
    }
  }
  if (any) {
    text.push_back(current ? 'B' : 'D');
    text += std::to_string(count);
  }
  return text;
}

bool load_model(const std::string& path, MarkerModel& model, std::string& error) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    error = "cannot open " + path;
    return false;
  }
  model = MarkerModel();
  if (!read_number(fs, "outer", model.outer, error) || !read_number(fs, "frame", model.frame, error) ||
      !read_number(fs, "arm", model.arm, error) || !read_number(fs, "notch_bright", model.notch_bright, error) ||
      !read_number(fs, "notch_gap", model.notch_gap, error)) {
    return false;
  }
  double notch_corner = 0;
  if (!read_number(fs, "notch_corner", notch_corner, error)) {
    return false;
  }
  model.notch_corner = static_cast<int>(std::lround(notch_corner));
  if (!(model.outer > model.frame && model.frame > 0 && model.arm > model.frame &&
        model.arm * 2 < model.outer && model.notch_corner >= 0 && model.notch_corner < 4 &&
        model.notch_bright > model.frame && model.notch_gap > 0 &&
        model.notch_bright + model.notch_gap < model.arm)) {
    error = "marker geometry does not describe four equal L corners with one notch";
    return false;
  }

  const CornerAxis axes[4] = {
      {{0, 0}, {1, 0}, {0, 1}},
      {{model.outer, 0}, {-1, 0}, {0, 1}},
      {{model.outer, model.outer}, {-1, 0}, {0, -1}},
      {{0, model.outer}, {1, 0}, {0, -1}},
  };
  for (int i = 0; i < 4; ++i) {
    model.corners[static_cast<std::size_t>(i)] = axes[i].origin;
  }

  auto add_strip = [&](int corner, int axis, double a, double b) {
    const CornerAxis& spec = axes[corner];
    const cv::Point2d along = axis == 0 ? spec.u : spec.v;
    const cv::Point2d width = axis == 0 ? spec.v : spec.u;
    const cv::Point2d p0 = spec.origin + along * a;
    const cv::Point2d p1 = spec.origin + along * b;
    const cv::Point2d p2 = p1 + width * model.frame;
    const cv::Point2d p3 = p0 + width * model.frame;
    Lamp lamp;
    lamp.x0 = std::min({p0.x, p1.x, p2.x, p3.x});
    lamp.y0 = std::min({p0.y, p1.y, p2.y, p3.y});
    lamp.x1 = std::max({p0.x, p1.x, p2.x, p3.x});
    lamp.y1 = std::max({p0.y, p1.y, p2.y, p3.y});
    if (!(lamp.x1 > lamp.x0 && lamp.y1 > lamp.y0)) {
      return;
    }
    lamp.quad = {{{lamp.x0, lamp.y0}, {lamp.x1, lamp.y0}, {lamp.x1, lamp.y1}, {lamp.x0, lamp.y1}}};
    model.lamps.push_back(lamp);
  };

  auto intervals = [&](int corner) {
    std::vector<std::pair<double, double>> spans;
    if (corner != model.notch_corner) {
      spans.emplace_back(0, model.arm);
      return spans;
    }
    spans.emplace_back(0, model.notch_bright);
    spans.emplace_back(model.notch_bright + model.notch_gap, model.arm);
    return spans;
  };

  auto add_gap = [&](int corner, int axis, double a, double b) {
    const std::size_t before = model.lamps.size();
    add_strip(corner, axis, a, b);
    if (model.lamps.size() == before) {
      return;
    }
    model.gaps.push_back(model.lamps.back());
    model.lamps.pop_back();
  };
  for (int corner = 0; corner < 4; ++corner) {
    for (const auto& span : intervals(corner)) {
      add_strip(corner, 0, span.first, span.second);
      add_strip(corner, 1, span.first, span.second);
    }
    if (corner == model.notch_corner) {
      const double a = model.notch_bright;
      const double b = model.notch_bright + model.notch_gap;
      add_gap(corner, 0, a, b);
      add_gap(corner, 1, a, b);
    }
  }

  auto make_l = [&](int corner, double length) {
    const CornerAxis& spec = axes[corner];
    const double w = model.frame;
    std::array<cv::Point2d, 6> poly = {
        spec.origin,
        spec.origin + spec.u * length,
        spec.origin + spec.u * length + spec.v * w,
        spec.origin + spec.u * w + spec.v * w,
        spec.origin + spec.u * w + spec.v * length,
        spec.origin + spec.v * length,
    };
    model.seeds.push_back(poly);
  };
  for (int corner = 0; corner < 4; ++corner) {
    if (corner == model.notch_corner) {
      make_l(corner, model.notch_bright);
    } else {
      make_l(corner, model.arm);
    }
  }

  auto inside = [&](double x, double y) {
    for (const Lamp& lamp : model.lamps) {
      if (x >= lamp.x0 && x < lamp.x1 && y >= lamp.y0 && y < lamp.y1) {
        return true;
      }
    }
    return false;
  };
  auto consider = [&](cv::Point2d a, cv::Point2d b, cv::Point2d outward) {
    const cv::Point2d mid = (a + b) * 0.5;
    const cv::Point2d out = mid + outward * 0.5;
    const cv::Point2d in = mid - outward * 0.5;
    if (inside(out.x, out.y) || !inside(in.x, in.y)) {
      return;
    }
    model.boundary.push_back({a, b, outward});
  };
  for (const Lamp& lamp : model.lamps) {
    consider({lamp.x0, lamp.y0}, {lamp.x1, lamp.y0}, {0, -1});
    consider({lamp.x1, lamp.y0}, {lamp.x1, lamp.y1}, {1, 0});
    consider({lamp.x1, lamp.y1}, {lamp.x0, lamp.y1}, {0, 1});
    consider({lamp.x0, lamp.y1}, {lamp.x0, lamp.y0}, {-1, 0});
  }

  const int steps = static_cast<int>(std::lround(model.outer));
  for (int edge = 0; edge < 4; ++edge) {
    int run_start = 0;
    bool current = false;
    bool any = false;
    auto emit = [&](int end, bool bright) {
      if (!any) {
        return;
      }
      Run run;
      run.bright = bright;
      run.a = run_start;
      run.b = end;
      const int index = static_cast<int>(model.runs.size());
      for (double c = run.a + 0.5; c < run.b; c += 1.0) {
        if (c < run.a + 0.0 || c > run.b) {
          continue;
        }
        cv::Point2d at;
        cv::Point2d inward;
        // The dark reference is the opening, not a step of one stroke.
        // A nearer point still sits in the lamp bloom and a bright run
        // looks dark.
        const double mid = model.outer * 0.5;
        if (edge == 0) {
          at = {c, model.frame * 0.5};
          inward = {c, mid};
        } else if (edge == 1) {
          at = {model.outer - model.frame * 0.5, c};
          inward = {mid, c};
        } else if (edge == 2) {
          at = {c, model.outer - model.frame * 0.5};
          inward = {c, mid};
        } else {
          at = {model.frame * 0.5, c};
          inward = {mid, c};
        }
        const bool hole = inward.x > model.frame && inward.x < model.outer - model.frame &&
                          inward.y > model.frame && inward.y < model.outer - model.frame;
        if (!hole) {
          continue;
        }
        model.samples.push_back({at, inward, index, c});
      }
      model.runs.push_back(run);
    };
    for (int s = 0; s < steps; ++s) {
      const double c = s + 0.5;
      const bool lit = edge == 0   ? model.frame_lit(c, model.frame * 0.5)
                       : edge == 1 ? model.frame_lit(model.outer - model.frame * 0.5, c)
                       : edge == 2 ? model.frame_lit(c, model.outer - model.frame * 0.5)
                                   : model.frame_lit(model.frame * 0.5, c);
      if (!any) {
        any = true;
        current = lit;
        run_start = s;
      } else if (lit != current) {
        emit(s, current);
        current = lit;
        run_start = s;
      }
    }
    emit(steps, current);
  }
  if (model.seeds.size() != 4 || model.samples.empty() || model.boundary.empty()) {
    error = "marker model produced no pattern";
    return false;
  }
  return true;
}

bool load_params(const std::string& path, DetectorParams& params, std::string& error) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    error = "cannot open " + path;
    return false;
  }
  params = DetectorParams();
  const cv::FileNode levels = fs["levels"];
  if (!levels.isSeq() || levels.empty()) {
    error = "missing levels";
    return false;
  }
  for (const auto& level : levels) {
    params.levels.push_back(static_cast<int>(std::lround(static_cast<double>(level))));
  }
  auto need = [&](const char* key, double& out) { return read_number(fs, key, out, error); };
  double max_polygons = 0;
  double min_samples = 0;
  double min_confirmed = 0;
  double lm_iters = 0;
  double min_edge_samples = 0;
  if (!need("segment_scale", params.segment_scale) || !need("approx_frac", params.approx_frac) ||
      !need("min_area", params.min_area) || !need("dedup_px", params.dedup_px) ||
      !need("max_polygons", max_polygons) || !need("reproj_frac", params.reproj_frac) ||
      !need("lit_ratio", params.lit_ratio) || !need("unlit_ratio", params.unlit_ratio) ||
      !need("min_contrast", params.min_contrast) || !need("run_margin", params.run_margin) ||
      !need("min_samples", min_samples) || !need("agree_frac", params.agree_frac) ||
      !need("min_confirmed_runs", min_confirmed) || !need("pose_agree_frac", params.pose_agree_frac) ||
      !need("rival_cost_ratio", params.rival_cost_ratio) ||
      !need("search_widths", params.search_widths) || !need("min_search_px", params.min_search_px) ||
      !need("sample_step", params.sample_step) || !need("lm_iters", lm_iters) ||
      !need("huber_px", params.huber_px) || !need("min_edge_frac", params.min_edge_frac) ||
      !need("min_edge_samples", min_edge_samples) || !need("min_side_span", params.min_side_span)) {
    return false;
  }
  params.max_polygons = static_cast<int>(std::lround(max_polygons));
  params.min_samples = static_cast<int>(std::lround(min_samples));
  params.min_confirmed_runs = static_cast<int>(std::lround(min_confirmed));
  params.lm_iters = static_cast<int>(std::lround(lm_iters));
  params.min_edge_samples = static_cast<int>(std::lround(min_edge_samples));
  if (!(params.segment_scale > 0 && params.segment_scale <= 1 && params.approx_frac > 0 &&
        params.min_area >= 0 && params.dedup_px > 0 && params.max_polygons > 0 && params.reproj_frac > 0 &&
        params.lit_ratio > params.unlit_ratio && params.unlit_ratio > 0 && params.lit_ratio < 1 &&
        params.min_contrast > 0 && params.run_margin >= 0 && params.min_samples > 0 &&
        params.agree_frac > 0.5 && params.agree_frac <= 1 && params.min_confirmed_runs > 0 &&
        params.pose_agree_frac > 0 && params.rival_cost_ratio > 1 && params.search_widths > 0 &&
        params.min_search_px > 0 &&
        params.sample_step > 0 && params.lm_iters > 0 && params.huber_px > 0 && params.min_edge_frac > 0 &&
        params.min_edge_frac <= 1 && params.min_edge_samples > 0 && params.min_side_span > 0 &&
        params.min_side_span <= 1)) {
    error = "detector parameters are not usable";
    return false;
  }
  return true;
}
