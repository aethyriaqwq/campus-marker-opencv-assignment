#include "PoseEstimator.h"

#include <iostream>

PoseEstimator::PoseEstimator(const cv::Mat& camera_matrix,
                             const cv::Mat& dist_coeffs)
    : camera_matrix_(camera_matrix.clone()),
      dist_coeffs_(dist_coeffs.clone()) {}

// 位姿解算
bool PoseEstimator::solve(const std::vector<cv::Point2f>& image_points,
                          const std::vector<cv::Point3f>& object_points) {
  if (image_points.size() < 4 || object_points.size() < 4) {
    std::cerr << "Not enough points for pose estimation." << std::endl;
    return false;
  }
  bool success = cv::solvePnP(object_points, image_points, camera_matrix_,
                              dist_coeffs_, rvec_, tvec_);
  return success;
}

// 绘制坐标轴
void PoseEstimator::drawAxis(cv::Mat& frame, float axis_length) {
  std::vector<cv::Point3f> axis_points = {
      cv::Point3f(0, 0, 0), cv::Point3f(axis_length, 0, 0),
      cv::Point3f(0, axis_length, 0), cv::Point3f(0, 0, axis_length)};
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(axis_points, rvec_, tvec_, camera_matrix_, dist_coeffs_,
                    image_points);

  cv::line(frame, image_points[0], image_points[1], cv::Scalar(0, 0, 255),
           3);  // X-axis in red
  cv::line(frame, image_points[0], image_points[2], cv::Scalar(0, 255, 0),
           3);  // Y-axis in green
  cv::line(frame, image_points[0], image_points[3], cv::Scalar(255, 0, 0),
           3);  // Z-axis in blue

  double distance_m = cv::norm(tvec_);
  double distance_mm = distance_m * 1000.0;  // Convert to millimeters
  cv::putText(frame, "Distance: " + std::to_string(distance_mm) + " mm",
              cv::Point(30, 80), cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(0, 255, 255), 2);
}