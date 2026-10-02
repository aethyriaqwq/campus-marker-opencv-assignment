#include "marker_tracker.hpp"

#include <opencv2/imgproc.hpp>

namespace marker {

// 构造：四个顶点各一路滤波器
Tracker::Tracker(const TrackerParams& params) : params_(params) {
    for (auto& filter : filters_) {
        filter = tracking::KalmanCV2D(params.filter_process_noise,
                                      params.filter_measurement_noise,
                                      params.filter_predict_steps);
    }
}

// 重算中心、宽高、角度
void Tracker::refreshGeometry() {
    const std::vector<cv::Point2f> points(smoothed_.corners.begin(),
                                          smoothed_.corners.end());
    smoothed_.center = (points[0] + points[1] + points[2] + points[3]) * 0.25F;
    const cv::RotatedRect box = cv::minAreaRect(points);
    smoothed_.width = box.size.width;
    smoothed_.height = box.size.height;
    smoothed_.angle_degrees = box.angle;
}

Marker Tracker::update(const Marker& input) {
    if (input.found) {
        // 目标重新出现时复位滤波器
        if (!active_) {
            for (auto& filter : filters_) {
                filter.reset();
            }
            active_ = true;
        }
        // 四个顶点分别滤波
        for (std::size_t i = 0; i < smoothed_.corners.size(); ++i) {
            smoothed_.corners[i] = filters_[i].update(input.corners[i]);
        }
        refreshGeometry();
        smoothed_.compactness = input.compactness;
        smoothed_.found = true;
        missed_ = 0;
        return smoothed_;
    }

    ++missed_;
    if (active_ && missed_ <= params_.max_missed_frames) {
        // 短暂丢失：用运动模型前推
        for (std::size_t i = 0; i < smoothed_.corners.size(); ++i) {
            smoothed_.corners[i] = filters_[i].advance(1);
        }
        refreshGeometry();
        smoothed_.found = true;
        return smoothed_;
    }

    // 持续丢失：返回未检测到
    active_ = false;
    return Marker{};
}

}  // namespace marker
