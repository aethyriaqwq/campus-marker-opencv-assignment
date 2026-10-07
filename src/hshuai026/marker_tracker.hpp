// 跨帧跟踪模块：对检测结果做卡尔曼滤波并处理短暂丢失。
#pragma once

#include "marker_detector.hpp"
#include "tracking_filter.hpp"

#include <array>

namespace marker {

// 跟踪参数
struct TrackerParams {
    double filter_process_noise = 0.5;      // 过程噪声 q
    double filter_measurement_noise = 9.0;  // 观测噪声 r
    int filter_predict_steps = 0;           // 延迟补偿帧数
    int max_missed_frames = 5;              // 判定丢失前允许的连续丢帧数
};

class Tracker {
public:
    explicit Tracker(const TrackerParams& params = TrackerParams());

    // 输入单帧检测结果，输出画面上使用的跟踪结果
    Marker update(const Marker& input);

    int missed() const { return missed_; }
    bool active() const { return active_; }

private:
    // 由四个顶点重算中心、宽高、角度
    void refreshGeometry();

    TrackerParams params_;
    std::array<tracking::KalmanCV2D, 4> filters_{};
    Marker smoothed_;
    int missed_ = 0;
    bool active_ = false;
};

}  // namespace marker
