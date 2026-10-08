#pragma once

#include <opencv2/core.hpp>

#include <array>

struct Detection {
  bool found = false;
  // Marker corners in drawing order: LT, RT, RB, LB. A measured corner is the
  // intersection of two outer edges; a side that is not visible keeps the
  // lamp-centroid model. Upright in the image, this is also image order.
  std::array<cv::Point2f, 4> corners{};
  // How the plate was fixed: 3 three solid lamps or a carried plate, 2 two lamps, 1 one lamp.
  int support = 0;
  float score = 0.f;
};

Detection detect_marker(const cv::Mat& bgr);
void draw_detection(cv::Mat& bgr, const Detection& detection);
