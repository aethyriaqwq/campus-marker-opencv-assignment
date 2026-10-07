#include "overlay.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <sstream>

namespace overlay {

namespace {
// 顶点显示名，顺序与 orderCorners 一致
const std::array<const char*, 4> kCornerLabels = {"LT", "RT", "RB", "LB"};
}

// 半透明底板
void drawInfoPanel(cv::Mat& image, const cv::Rect& area, double alpha) {
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, area, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image);
}

// 检测结果可视化
void drawMarker(cv::Mat& image, const marker::Marker& marker, bool holding) {
    drawInfoPanel(image, cv::Rect(0, 0, 660, marker.found ? 232 : 76));
    if (!marker.found) {
        cv::putText(image, "NO TARGET", cv::Point(20, 52), cv::FONT_HERSHEY_SIMPLEX,
                    1.3, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
        return;
    }

    // 框的颜色：预测值为橙色，正常为绿色
    const cv::Scalar color =
        holding ? cv::Scalar(0, 200, 255) : cv::Scalar(0, 255, 0);

    // 画四条边
    for (int i = 0; i < 4; ++i) {
        cv::line(image, marker::toPixel(marker.corners[i]),
                 marker::toPixel(marker.corners[(i + 1) % 4]), color, 3, cv::LINE_AA);
    }

    // 画顶点与顶点坐标文字
    for (int i = 0; i < 4; ++i) {
        const cv::Point p = marker::toPixel(marker.corners[i]);
        cv::circle(image, p, 7, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        std::ostringstream point_text;
        point_text << kCornerLabels[i] << "(" << p.x << "," << p.y << ")";
        const cv::Point at = p + cv::Point(12, (i < 2) ? -14 : 26);
        cv::putText(image, point_text.str(), at, cv::FONT_HERSHEY_SIMPLEX, 0.75,
                    cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::putText(image, point_text.str(), at, cv::FONT_HERSHEY_SIMPLEX, 0.75,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }

    // 第一行：状态与中心坐标
    std::ostringstream line;
    line << (holding ? "TARGET (predict)" : "TARGET") << "   center=("
         << cvRound(marker.center.x) << "," << cvRound(marker.center.y) << ")";
    cv::putText(image, line.str(), cv::Point(20, 52), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                color, 3, cv::LINE_AA);

    // 第二行：尺寸与角度
    line.str("");
    line << "size=" << cvRound(marker.width) << "x" << cvRound(marker.height)
         << " px   angle=" << cv::format("%.1f", marker.angle_degrees) << " deg";
    cv::putText(image, line.str(), cv::Point(20, 96), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // 后四行：四个顶点坐标
    for (int i = 0; i < 4; ++i) {
        std::ostringstream corner;
        corner << kCornerLabels[i] << " = (" << cvRound(marker.corners[i].x) << ", "
               << cvRound(marker.corners[i].y) << ")";
        cv::putText(image, corner.str(), cv::Point(24, 132 + i * 26),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2,
                    cv::LINE_AA);
    }
}

// 位姿数值可视化
void drawPose(cv::Mat& image, double x_mm, double y_mm, double z_mm, double distance_mm,
              double rx_deg, double ry_deg, double rz_deg, double reproj_px,
              double marker_size_mm, int top) {
    drawInfoPanel(image, cv::Rect(0, top, 760, 190));
    std::ostringstream text;

    // 位置
    text << "POSITION  X=" << cvRound(x_mm) << " mm  Y=" << cvRound(y_mm)
         << " mm  Z=" << cvRound(z_mm) << " mm";
    cv::putText(image, text.str(), cv::Point(20, top + 50), cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // 距离
    text.str("");
    text << "DISTANCE=" << cvRound(distance_mm) << " mm   ("
         << cv::format("%.2f", distance_mm / 1000.0) << " m)";
    cv::putText(image, text.str(), cv::Point(20, top + 92), cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

    // 姿态
    text.str("");
    text << "ROTATION  rx=" << cv::format("%.1f", rx_deg)
         << " deg  ry=" << cv::format("%.1f", ry_deg)
         << " deg  rz=" << cv::format("%.1f", rz_deg) << " deg";
    cv::putText(image, text.str(), cv::Point(20, top + 134),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(200, 255, 200), 2,
                cv::LINE_AA);

    // 重投影误差
    text.str("");
    text << "reproj error=" << cv::format("%.2f", reproj_px)
         << " px   marker=" << cvRound(marker_size_mm) << "x"
         << cvRound(marker_size_mm) << " mm";
    cv::putText(image, text.str(), cv::Point(20, top + 172),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2,
                cv::LINE_AA);
}

// 坐标轴字母
void drawAxisLabels(cv::Mat& image, const std::vector<cv::Point2f>& tips) {
    const std::array<const char*, 3> names = {"X", "Y", "Z"};
    const std::array<cv::Scalar, 3> colors = {cv::Scalar(0, 0, 255),
                                              cv::Scalar(0, 255, 0),
                                              cv::Scalar(255, 0, 0)};
    for (std::size_t i = 0; i < tips.size() && i < names.size(); ++i) {
        cv::putText(image, names[i], marker::toPixel(tips[i]), cv::FONT_HERSHEY_SIMPLEX,
                    1.1, colors[i], 3, cv::LINE_AA);
    }
}

}  // namespace overlay
