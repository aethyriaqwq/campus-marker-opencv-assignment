#include "calibrate.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

cv::Ptr<cv::SimpleBlobDetector> circle_detector() {
  cv::SimpleBlobDetector::Params params;
  params.filterByColor = true;
  params.blobColor = 0;
  params.filterByArea = true;
  params.minArea = 20;
  params.maxArea = 20000;
  params.filterByCircularity = true;
  params.minCircularity = 0.6f;
  params.filterByConvexity = false;
  params.filterByInertia = false;
  return cv::SimpleBlobDetector::create(params);
}

}  // namespace

bool calibrate_camera(const std::string& video_path, const std::string& yaml_path) {
  cv::VideoCapture capture(video_path);
  if (!capture.isOpened()) {
    std::cerr << "cannot open " << video_path << "\n";
    return false;
  }
  // The board printed in the video is a 7 by 7 symmetric circle grid with a corner chamfer.
  // Neighboring frames repeat one pose, so only every 25th frame is used.
  constexpr int kStride = 25;
  const cv::Size pattern(7, 7);
  const int grid_flags = cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING;
  auto detector = circle_detector();
  std::vector<cv::Point3f> model;
  model.reserve(static_cast<size_t>(pattern.area()));
  for (int row = 0; row < pattern.height; ++row) {
    for (int col = 0; col < pattern.width; ++col) {
      model.emplace_back(static_cast<float>(col), static_cast<float>(row), 0.f);
    }
  }
  std::vector<std::vector<cv::Point3f>> objects;
  std::vector<std::vector<cv::Point2f>> images;
  cv::Mat frame;
  cv::Size image_size;
  int index = 0;
  while (capture.read(frame)) {
    if (frame.empty()) {
      break;
    }
    image_size = frame.size();
    if (index % kStride == 0) {
      cv::Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      std::vector<cv::Point2f> centers;
      if (cv::findCirclesGrid(gray, pattern, centers, grid_flags, detector) &&
          static_cast<int>(centers.size()) == pattern.area()) {
        objects.push_back(model);
        images.push_back(std::move(centers));
      }
    }
    ++index;
  }
  if (objects.size() < 10 || image_size.width <= 0) {
    std::cerr << "too few circle-grid views\n";
    return false;
  }
  cv::Mat camera = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  cv::Mat std_int;
  cv::Mat std_ext;
  cv::Mat per_view;
  // The sixth-order term changes reprojection by far less than its own uncertainty.
  const int flags = cv::CALIB_FIX_K3;
  const double rms = cv::calibrateCamera(
      objects, images, image_size, camera, dist, rvecs, tvecs, std_int, std_ext, per_view, flags,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 1e-9));
  cv::FileStorage out(yaml_path, cv::FileStorage::WRITE);
  if (!out.isOpened()) {
    std::cerr << "cannot write " << yaml_path << "\n";
    return false;
  }
  out << "image_width" << image_size.width;
  out << "image_height" << image_size.height;
  out << "camera_matrix" << camera;
  out << "distortion_coefficients" << dist;
  out << "rms" << rms;
  out << "views" << static_cast<int>(images.size());
  out << "pattern_cols" << pattern.width;
  out << "pattern_rows" << pattern.height;
  out << "object_unit" << "circle-center spacing";
  out << "board_label" << "HC300-15-7x7";
  out.release();
  std::cout << "views " << images.size() << "\n"
            << "rms " << rms << "\n"
            << "camera\n"
            << camera << "\n"
            << "distortion " << dist << "\n";
  return std::isfinite(rms);
}
