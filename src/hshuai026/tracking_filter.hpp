// 匀速卡尔曼滤波（CV 模型）：状态 [x, y, vx, vy]，观测为位置。
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cmath>

namespace tracking {

class KalmanCV2D {
public:
    // q 过程噪声，r 观测噪声方差，predict_steps 为输出前的额外预测帧数
    KalmanCV2D(double q = 0.5, double r = 9.0, int predict_steps = 0)
        : q_(q), r_(r), predict_steps_(predict_steps) {}

    // 复位滤波器
    void reset() {
        initialized_ = false;
        state_ = {0.0, 0.0, 0.0, 0.0};
        for (auto& row : covariance_) {
            row.fill(0.0);
        }
    }

    bool initialized() const { return initialized_; }

    // 用一帧检测值更新，返回滤波后的位置
    cv::Point2f update(const cv::Point2f& measurement) {
        if (!initialized_) {
            state_[0] = measurement.x;
            state_[1] = measurement.y;
            for (int i = 0; i < 4; ++i) {
                covariance_[i][i] = (i < 2) ? 100.0 : 1000.0;
            }
            initialized_ = true;
            return current();
        }
        predict();
        correct(measurement);
        return current();
    }

    // 无检测时按运动模型前推指定帧数
    cv::Point2f advance(int steps = 1) {
        for (int i = 0; i < steps; ++i) {
            predict();
        }
        return current();
    }

    // 当前估计位置
    cv::Point2f current() const {
        return cv::Point2f(static_cast<float>(state_[0] + predict_steps_ * state_[2]),
                           static_cast<float>(state_[1] + predict_steps_ * state_[3]));
    }

private:
    // 状态预测
    void predict() {
        state_[0] += state_[2];
        state_[1] += state_[3];

        // f(i,k) 为状态转移矩阵元素
        auto f = [](int i, int k) {
            if (i == k) {
                return 1.0;
            }
            return (i < 2 && k == i + 2) ? 1.0 : 0.0;
        };

        // P = F * P
        double fp[4][4] = {};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 4; ++k) {
                    sum += f(i, k) * covariance_[k][j];
                }
                fp[i][j] = sum;
            }
        }

        // P = (F * P) * F^T + Q
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 4; ++k) {
                    sum += fp[i][k] * f(j, k);
                }
                covariance_[i][j] = sum + ((i == j) ? q_ : 0.0);
            }
        }
    }

    // 观测更新
    void correct(const cv::Point2f& measurement) {
        // S = H * P * H^T + R（H 只取位置两行）
        const double s00 = covariance_[0][0] + r_;
        const double s01 = covariance_[0][1];
        const double s10 = covariance_[1][0];
        const double s11 = covariance_[1][1] + r_;
        const double det = s00 * s11 - s01 * s10;
        if (std::abs(det) < 1e-12) {
            return;
        }
        const double inv00 = s11 / det;
        const double inv01 = -s01 / det;
        const double inv10 = -s10 / det;
        const double inv11 = s00 / det;

        // K = P * H^T * S^-1
        double k[4][2];
        for (int i = 0; i < 4; ++i) {
            k[i][0] = covariance_[i][0] * inv00 + covariance_[i][1] * inv10;
            k[i][1] = covariance_[i][0] * inv01 + covariance_[i][1] * inv11;
        }

        // 状态更新
        const double residual_x = measurement.x - state_[0];
        const double residual_y = measurement.y - state_[1];
        for (int i = 0; i < 4; ++i) {
            state_[i] += k[i][0] * residual_x + k[i][1] * residual_y;
        }

        // P = (I - K * H) * P
        double updated[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                updated[i][j] = covariance_[i][j] -
                                k[i][0] * covariance_[0][j] -
                                k[i][1] * covariance_[1][j];
            }
        }

        // 对称化
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                covariance_[i][j] = 0.5 * (updated[i][j] + updated[j][i]);
            }
        }
    }

    double q_ = 0.5;
    double r_ = 9.0;
    int predict_steps_ = 0;
    bool initialized_ = false;
    std::array<double, 4> state_{};
    std::array<std::array<double, 4>, 4> covariance_{};
};

}  // namespace tracking
