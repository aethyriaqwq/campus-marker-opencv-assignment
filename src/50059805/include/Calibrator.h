#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/*
 * 相机标定类，用于从棋盘格图像中提取角点并进行相机标定
*/

class Calibrator {
 public:
  Calibrator();
  bool extractCorners(const cv::Mat& frame, std::vector<cv::Point2f>& corners);

  double calibrate(const std::vector<std::vector<cv::Point2f> >& image_points,
                   cv::Size board_size, float square_size, cv::Size image_size);
  
  void saveParams(const std::string& filename);
  bool loadParams(const std::string& filename);

  cv::Mat getCameraMatrix() const { return camera_matrix_; }
  cv::Mat getDistCoeffs() const { return dist_coeffs_; }

 private:
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
};