#include "MarkerDetector.h"

#include <string>

MarkerDetector::MarkerDetector() {}

// 预处理图像：灰度化、二值化、形态学操作
cv::Mat MarkerDetector::preprocess(const cv::Mat& frame) {
  if (frame.empty()) {
    return cv::Mat();
  }

  cv::Mat gray;
  // 1. 预处理：转灰度
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

  // 2. 二值化：因为Marker是发光的，提取高亮区域
  // 阈值(200)需要根据实际视频调整
  cv::inRange(gray, cv::Scalar(200), cv::Scalar(255), gray);

  // 3. 形态学操作：闭运算将四个分离的L型连接成一个整体，再开运算去除噪点
  // 补充：这里连接效果不好，所以改为只进行开运算，去除噪点
  cv::Mat morphed;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  // cv::morphologyEx(gray, gray, cv::MORPH_CLOSE, kernel);
  cv::morphologyEx(gray, morphed, cv::MORPH_OPEN, kernel);

  return morphed;
}


// 检测Marker
MarkerResult MarkerDetector::detect(const cv::Mat& frame) {
  MarkerResult result;

  if (frame.empty()) {
    return result;
  }

  // (1) 预处理：灰度化、二值化、形态学操作
  cv::Mat preprocessed_img = preprocess(frame);
  if (preprocessed_img.empty()) {
    return result;
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(preprocessed_img, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::Point2i> all_l_points;

  double min_area = 100.0;  // 最小面积阈值，根据实际情况调整

  // 4. 寻找轮廓
  // 5. 几何筛选：多边形拟合，判断是否为四边形
  for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    if (area < min_area) {
      continue;  // 面积太小，忽略
    }

    cv::Rect rect = cv::boundingRect(contour);

    float aspect_ratio = static_cast<float>(rect.width) / rect.height;

    if (aspect_ratio < 0.5 || aspect_ratio > 2.0) {
      continue;  // 长宽比不符合要求，忽略
    }

    for (const auto& point : contour) {
      all_l_points.push_back(point);
    }
  }
  if (all_l_points.size() > 20) {
    cv::Rect bouding_box = cv::boundingRect(all_l_points);

    cv::RotatedRect rotated_rect = cv::minAreaRect(all_l_points);

    cv::Point2f rect_points[4];
    rotated_rect.points(rect_points);

    result.detected_ = true;
    result.bounding_box_ = bouding_box;

    std::vector<cv::Point2f> pts(rect_points, rect_points + 4);
    result.points_ = sortPoints(pts);
  }
  return result;
}

// 将四个点按顺时针排序，左上、右上、右下、左下
std::vector<cv::Point2f> MarkerDetector::sortPoints(
    const std::vector<cv::Point2f>& pts) {
  std::vector<cv::Point2f> sorted(4);
  cv::Point2f center(0, 0);

  for (const auto& pt : pts) {
    center += pt;
  }
  center *= 0.25f;

  for (const auto& pt : pts) {
    if (pt.x < center.x && pt.y < center.y) {
      sorted[0] = pt;  // 左上
    } else if (pt.x > center.x && pt.y < center.y) {
      sorted[1] = pt;  // 右上
    } else if (pt.x > center.x && pt.y > center.y) {
      sorted[2] = pt;  // 右下
    } else if (pt.x < center.x && pt.y > center.y) {
      sorted[3] = pt;  // 左下
    }
  }
  return sorted;
}
