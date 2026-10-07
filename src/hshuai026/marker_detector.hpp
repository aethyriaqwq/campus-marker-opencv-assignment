// 标志物检测模块：单帧输入，输出外框与四个顶点（LT/RT/RB/LB）。
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <vector>

namespace marker {

// 检测参数
struct DetectionParams {
    int bright_threshold = 200;     // 高亮阈值
    int merge_kernel = 51;          // 膨胀核边长
    double min_area = 1500.0;       // 候选面积下限
    double max_area = 300000.0;     // 候选面积上限
    double min_side = 80.0;         // 候选短边下限
    double max_side = 900.0;        // 候选长边上限
    double max_aspect = 2.0;        // 长宽比上限
    double min_compactness = 0.30;  // 面积 / 外接矩形面积下限
};

// 检测结果
struct Marker {
    bool found = false;                    // 本帧是否检测到
    std::array<cv::Point2f, 4> corners{};  // 四顶点：LT、RT、RB、LB
    cv::Point2f center{};                  // 中心
    float width = 0.0F;                    // 外框宽
    float height = 0.0F;                   // 外框高
    float angle_degrees = 0.0F;            // 外框角度
    double compactness = 0.0;              // 紧凑度
};

// 中间结果，用于调试
struct DetectionDebug {
    cv::Mat mask;          // 阈值 + 开运算后的掩膜
    cv::Mat merged;        // 膨胀合并后的掩膜
    int raw_candidates = 0;  // 通过几何筛选的候选数
};

cv::Point toPixel(const cv::Point2f& point);

// 把四个顶点整理成 LT、RT、RB、LB
std::array<cv::Point2f, 4> orderCorners(const std::vector<cv::Point2f>& points);

// 单帧检测
Marker detect(const cv::Mat& frame, const DetectionParams& params = DetectionParams(),
              DetectionDebug* debug = nullptr);

}  // namespace marker
