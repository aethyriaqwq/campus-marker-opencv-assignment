#pragma once
#include <opencv2/opencv.hpp>
#include <array>
#include <vector>

namespace marker {
using Quad = std::array<cv::Point2f, 4>;
struct Detection {
    Quad corners; // image-relative LT, RT, RB, LB, in original input pixels
    float score = 0;
    int id = -1;
};
struct Config {
    int workWidth = 960; // no upscaling, zero means original resolution
    int threshold = 180;
    float minScore = 0.64f;
    float smoothing = 0.75f; // weight of current observation
};
cv::Mat makeTemplate(int size = 160);
Quad orderCorners(Quad points);
class Detector {
public:
    explicit Detector(Config config = {});
    std::vector<Detection> process(const cv::Mat& frame);
    void reset();
private:
    Config config_;
    cv::Mat template_;
    std::vector<Detection> previous_;
    cv::Size lastSize_;
    int nextId_ = 0;
};
void draw(cv::Mat& frame, const std::vector<Detection>& detections, int frameIndex);
} // namespace marker
