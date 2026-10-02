// 可视化模块：把检测与位姿结果画到画面上。
#pragma once

#include "marker_detector.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace overlay {

// 画半透明底板
void drawInfoPanel(cv::Mat& image, const cv::Rect& area, double alpha = 0.45);

// 画外框、四个顶点、顶点像素坐标与中心坐标
void drawMarker(cv::Mat& image, const marker::Marker& marker, bool holding);

// 画位姿数值（位置、距离、姿态、重投影误差）
void drawPose(cv::Mat& image, double x_mm, double y_mm, double z_mm, double distance_mm,
              double rx_deg, double ry_deg, double rz_deg, double reproj_px,
              double marker_size_mm, int top);

// 在坐标轴末端标注 X / Y / Z
void drawAxisLabels(cv::Mat& image, const std::vector<cv::Point2f>& tips);

}  // namespace overlay
