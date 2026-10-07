// 位姿估计模块：用标定参数与标志物尺寸做 PnP，求标志物在相机坐标系下的位姿。
#pragma once

#include "marker_detector.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

namespace pose {

// 相机内参与畸变系数
struct CameraIntrinsics {
    cv::Mat camera_matrix;  // 3x3 内参矩阵
    cv::Mat dist_coeffs;    // 1x5 畸变系数

    // 从 YAML 读取（calib_dots 的输出）
    bool load(const std::string& yaml_path);
    bool valid() const { return !camera_matrix.empty(); }
};

// 单帧位姿结果
struct Result {
    bool valid = false;                  // 本帧是否解算成功
    cv::Mat rvec;                        // 旋转向量
    cv::Mat tvec;                        // 平移向量（mm）
    cv::Point3d position_mm{};           // tvec：标志物中心在相机坐标系中的位置
    std::array<double, 3> euler_deg{};   // 姿态角 rx / ry / rz（度）
    double distance_mm = 0.0;            // 到相机光心的距离
    double reprojection_error_px = 0.0;  // 该帧重投影误差
};

// 标志物平面模型点：原点在正面中心，X 向右、Y 向下、Z 指向相机，单位 mm
std::vector<cv::Point3f> markerModel(float size_mm);

// 求解一帧的位姿
Result solve(const std::array<cv::Point2f, 4>& corners, const CameraIntrinsics& camera,
             float size_mm);

}  // namespace pose
