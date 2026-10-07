#include "Calibrator.h"

Calibrator::Calibrator() {}

// 提取棋盘格角点
bool Calibrator::extractCorners(const cv::Mat& frame,
                                std::vector<cv::Point2f>& corners) {
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  bool found = cv::findCirclesGrid(gray, cv::Size(7, 7), corners,
                                   cv::CALIB_CB_SYMMETRIC_GRID);

  if (!found) {
    cv::Mat inv_gray;
    cv::bitwise_not(gray, inv_gray);
    found = cv::findCirclesGrid(inv_gray, cv::Size(7, 7), corners,
                                cv::CALIB_CB_SYMMETRIC_GRID);
  }
  return found;
}

// 使用提取的角点进行相机标定
double Calibrator::calibrate(
    const std::vector<std::vector<cv::Point2f> >& image_points,
    cv::Size board_size, float square_size, cv::Size image_size) {
  std::vector<std::vector<cv::Point3f> > object_points;
  std::vector<cv::Point3f> obj;
  for (int i = 0; i < board_size.height; i++) {
    for (int j = 0; j < board_size.width; j++) {
      obj.push_back(cv::Point3f(j * square_size, i * square_size, 0));
    }
  }

  for (size_t i = 0; i < image_points.size(); i++) {
    object_points.push_back(obj);
  }

  cv::Mat rvecs, tvecs;
  double ret = cv::calibrateCamera(object_points, image_points, image_size,
                                   camera_matrix_, dist_coeffs_, rvecs, tvecs);
  std::cout << "Calibration Reprojection Error: " << ret << " pixels"
            << std::endl;
  return ret;
}

// 保存相机参数到文件
void Calibrator::saveParams(const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  fs << "camera_matrix" << camera_matrix_;
  fs << "dist_coeffs" << dist_coeffs_;
  fs.release();
  std::cout << "Params saved to " << filename << std::endl;
}

// 从文件加载相机参数
bool Calibrator::loadParams(const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "Failed to open " << filename << std::endl;
    return false;
  }
  fs["camera_matrix"] >> camera_matrix_;
  fs["dist_coeffs"] >> dist_coeffs_;
  fs.release();
  std::cout << "Params loaded from " << filename << std::endl;
  return true;
}