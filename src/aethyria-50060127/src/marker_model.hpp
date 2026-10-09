#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

// Luminous marker. The drawing is a square frame. Each corner is an L of the
// same arm length. Three arms are solid. One arm pair is notched: bright,
// dark, bright, and the dark gap is what fixes orientation. The two bright
// pieces of a notched arm are parts of that same L.

struct Lamp {
  double x0 = 0;
  double y0 = 0;
  double x1 = 0;
  double y1 = 0;
  std::array<cv::Point2d, 4> quad{};
};

struct Run {
  bool bright = false;
  double a = 0;
  double b = 0;
};

struct Sample {
  cv::Point2d at;
  cv::Point2d inward;
  int run = 0;
  // Drawing coordinate along the edge. A sample is in the run interior when
  // it sits at least run_margin inside that run's [a, b).
  double coord = 0;
};

struct Boundary {
  cv::Point2d a;
  cv::Point2d b;
  cv::Point2d outward;
};

struct MarkerModel {
  double outer = 0;
  double frame = 0;
  double arm = 0;
  double notch_bright = 0;
  double notch_gap = 0;
  int notch_corner = 1;
  std::array<cv::Point2d, 4> corners{};
  std::vector<Lamp> lamps;
  // Dark rectangles of the notched L. They belong to that L; filling them
  // would make every corner a solid L of the same size.
  std::vector<Lamp> gaps;
  std::vector<std::array<cv::Point2d, 6>> seeds;
  std::vector<Run> runs;
  std::vector<Sample> samples;
  std::vector<Boundary> boundary;

  bool arm_lit(int corner, double distance) const;
  bool frame_lit(double x, double y) const;
  // Edge 0 top, 1 right, 2 bottom, 3 left, each read from the low coordinate.
  std::string edge_runs(int edge) const;
};

struct DetectorParams {
  std::vector<int> levels;
  double segment_scale = 1;
  double approx_frac = 0;
  double min_area = 0;
  double dedup_px = 0;
  int max_polygons = 0;
  double reproj_frac = 0;
  double lit_ratio = 0;
  double unlit_ratio = 0;
  double min_contrast = 0;
  double run_margin = 0;
  int min_samples = 0;
  double agree_frac = 0;
  int min_confirmed_runs = 0;
  double pose_agree_frac = 0;
  double rival_cost_ratio = 0;
  double search_widths = 0;
  double min_search_px = 0;
  double sample_step = 0;
  int lm_iters = 0;
  double huber_px = 0;
  double min_edge_frac = 0;
  int min_edge_samples = 0;
  // Inliers on a side must span this fraction of the outer side. A shorter
  // piece does not locate the far corner.
  double min_side_span = 0;
};

bool load_model(const std::string& path, MarkerModel& model, std::string& error);
bool load_params(const std::string& path, DetectorParams& params, std::string& error);
