#include "marker_detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace marker {

// 浮点坐标转整数像素
cv::Point toPixel(const cv::Point2f& point) {
    return {cvRound(point.x), cvRound(point.y)};
}

// 四个顶点排序为 LT、RT、RB、LB
std::array<cv::Point2f, 4> orderCorners(const std::vector<cv::Point2f>& points) {
    if (points.size() != 4) {
        throw std::invalid_argument("orderCorners 需要四个点");
    }
    std::vector<cv::Point2f> sorted = points;
    std::sort(sorted.begin(), sorted.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    std::vector<cv::Point2f> left(sorted.begin(), sorted.begin() + 2);
    std::vector<cv::Point2f> right(sorted.begin() + 2, sorted.end());
    auto by_y = [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; };
    std::sort(left.begin(), left.end(), by_y);
    std::sort(right.begin(), right.end(), by_y);
    return {left[0], right[0], right[1], left[1]};
}

Marker detect(const cv::Mat& frame, const DetectionParams& params,
              DetectionDebug* debug) {
    Marker result;
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // 高亮阈值
    cv::Mat mask;
    cv::threshold(gray, mask, params.bright_threshold, 255, cv::THRESH_BINARY);

    // 开运算去噪
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    // 膨胀合并灯块
    cv::Mat merged;
    cv::dilate(mask, merged,
               cv::getStructuringElement(cv::MORPH_RECT,
                                         cv::Size(params.merge_kernel,
                                                  params.merge_kernel)));

    if (debug != nullptr) {
        debug->mask = mask.clone();
        debug->merged = merged.clone();
    }

    // 提取外轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(merged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 按面积、边长、长宽比、紧凑度筛选候选
    double best_area = 0.0;
    cv::Rect best_region;
    int candidate_count = 0;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < params.min_area || area > params.max_area) {
            continue;
        }
        const cv::RotatedRect box = cv::minAreaRect(contour);
        const double long_side = std::max(box.size.width, box.size.height);
        const double short_side = std::min(box.size.width, box.size.height);
        if (short_side < params.min_side || long_side > params.max_side ||
            short_side <= 0.0 || long_side / short_side > params.max_aspect) {
            continue;
        }
        if (area / (box.size.width * box.size.height) < params.min_compactness) {
            continue;
        }
        ++candidate_count;
        if (area > best_area) {
            best_area = area;
            best_region = cv::boundingRect(contour);
        }
    }
    if (debug != nullptr) {
        debug->raw_candidates = candidate_count;
    }
    if (best_area <= 0.0) {
        return result;
    }

    // 取候选区域内的原始亮点
    const cv::Rect clipped = best_region & cv::Rect(0, 0, mask.cols, mask.rows);
    cv::Mat region = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::rectangle(region, clipped, 255, cv::FILLED);
    cv::Mat target;
    cv::bitwise_and(mask, region, target);
    std::vector<cv::Point> points;
    cv::findNonZero(target, points);
    if (points.size() < 50) {
        return result;
    }

    // 凸包 + 最小外接旋转矩形
    std::vector<cv::Point> hull;
    cv::convexHull(points, hull);
    const cv::RotatedRect box = cv::minAreaRect(hull);
    cv::Point2f raw_corners[4];
    box.points(raw_corners);

    // 填入结果
    result.corners =
        orderCorners(std::vector<cv::Point2f>(raw_corners, raw_corners + 4));
    result.center = box.center;
    result.width = box.size.width;
    result.height = box.size.height;
    result.angle_degrees = box.angle;
    result.compactness = static_cast<double>(points.size()) /
                         (box.size.width * box.size.height);
    result.found = true;
    return result;
}

}  // namespace marker
