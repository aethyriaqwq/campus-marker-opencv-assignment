// 主程序：标志物检测 + 跟踪 +（可选）位姿估计，输出视频与日志。
//
// 用法: campus_marker [输入视频] [输出目录] [标定yaml] [标志物边长mm]
//   给了标定 yaml 就同时输出位姿（坐标轴 + mm 数值），否则只做检测。

#include "marker_detector.hpp"
#include "marker_tracker.hpp"
#include "overlay.hpp"
#include "pose_estimator.hpp"
#include "video_utils.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kOverviewStep = 150;   // 概览截图间隔
constexpr int kSequenceStart = 250;  // 连续帧截图起点
constexpr int kSequenceEnd = 280;    // 连续帧截图终点
constexpr int kReportStep = 25;      // 进度打印间隔

// 保存半尺寸 JPEG 截图
void saveEvidence(const cv::Mat& image, const fs::path& path) {
    cv::Mat half;
    cv::resize(image, half, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::imwrite(path.string(), half, {cv::IMWRITE_JPEG_QUALITY, 90});
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path input_path = video::resolveInputPath(
        argc > 1 ? fs::path(argv[1]) : fs::path("data/raw/marker_video.avi"));
    const fs::path output_dir = video::resolveOutputDir(
        argc > 2 ? fs::path(argv[2]) : fs::path("src/hshuai026/results"), input_path);
    const fs::path calib_path = argc > 3 ? fs::path(argv[3]) : fs::path();
    const float marker_size_mm = argc > 4 ? static_cast<float>(std::atof(argv[4])) : 80.0F;

    try {
        if (!fs::is_regular_file(input_path)) {
            throw std::runtime_error("输入视频不存在: " + input_path.string());
        }
        cv::VideoCapture capture;
        if (!video::openCapture(capture, input_path.string())) {
            throw std::runtime_error("无法读取输入视频: " + input_path.string());
        }
        fs::create_directories(output_dir);

        // 可选：加载标定参数，用于位姿估计
        pose::CameraIntrinsics camera;
        bool pose_enabled = false;
        if (!calib_path.empty()) {
            pose_enabled = camera.load(calib_path.string());
            if (!pose_enabled) {
                std::cerr << "警告: 读不到标定文件 " << calib_path.string()
                          << "，本次只输出检测结果\n";
            } else {
                std::cout << "已加载标定: " << calib_path.string()
                          << "（同时输出位姿，单位 mm）\n";
            }
        }

        double fps = capture.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 0.0) {
            fps = 30.0;
        }
        const fs::path video_path =
            output_dir / (input_path.stem().string() + "_result.mp4");
        const fs::path csv_path =
            output_dir / (input_path.stem().string() + "_result.csv");

        cv::VideoWriter writer;
        std::ofstream csv(csv_path);
        csv << "frame,found,cx,cy,width,height,angle,pose_ok,x_mm,y_mm,z_mm,"
               "distance_mm,rx_deg,ry_deg,rz_deg,reproj_px\n";

        marker::Tracker tracker;
        cv::Mat frame;
        int index = 0;
        int detected = 0;
        int solved = 0;
        std::vector<double> distances;
        while (capture.read(frame)) {
            if (frame.empty()) {
                throw std::runtime_error("视频中出现空帧");
            }

            // 检测与跟踪
            marker::DetectionDebug debug;
            const marker::Marker raw =
                marker::detect(frame, marker::DetectionParams(), &debug);
            const bool raw_found = raw.found;
            const marker::Marker shown = tracker.update(raw);
            const bool holding = shown.found && !raw_found;

            // 位姿估计（可选）
            pose::Result pose_result;
            if (pose_enabled && shown.found) {
                pose_result = pose::solve(shown.corners, camera, marker_size_mm);
            }

            // 绘制
            cv::Mat annotated = frame.clone();
            overlay::drawMarker(annotated, shown, holding);
            if (pose_result.valid) {
                // 坐标轴（红 X / 绿 Y / 蓝 Z），轴长为标志物边长
                cv::drawFrameAxes(annotated, camera.camera_matrix, camera.dist_coeffs,
                                  pose_result.rvec, pose_result.tvec, marker_size_mm);
                const std::vector<cv::Point3f> axis_tips = {
                    {marker_size_mm / 2.0F, 0.0F, 0.0F},
                    {0.0F, marker_size_mm / 2.0F, 0.0F},
                    {0.0F, 0.0F, marker_size_mm / 2.0F}};
                std::vector<cv::Point2f> tips;
                cv::projectPoints(axis_tips, pose_result.rvec, pose_result.tvec,
                                  camera.camera_matrix, camera.dist_coeffs, tips);
                overlay::drawAxisLabels(annotated, tips);

                // 位姿数值
                overlay::drawPose(annotated, pose_result.position_mm.x,
                                  pose_result.position_mm.y, pose_result.position_mm.z,
                                  pose_result.distance_mm, pose_result.euler_deg[0],
                                  pose_result.euler_deg[1], pose_result.euler_deg[2],
                                  pose_result.reprojection_error_px, marker_size_mm, 240);
                distances.push_back(pose_result.distance_mm);
                ++solved;
            }

            // 写视频
            if (!writer.isOpened() &&
                !video::openWriter(writer, video_path, fps, frame.size())) {
                throw std::runtime_error("无法创建输出视频: " + video_path.string());
            }
            writer.write(annotated);

            // 截图
            if (index % kOverviewStep == 0) {
                std::ostringstream name;
                name << "overview_" << std::setw(5) << std::setfill('0') << index
                     << ".jpg";
                saveEvidence(annotated, output_dir / name.str());
            }
            if (index >= kSequenceStart && index < kSequenceEnd) {
                std::ostringstream name;
                name << "sequence_" << std::setw(5) << std::setfill('0') << index
                     << ".jpg";
                saveEvidence(annotated, output_dir / name.str());
            }

            // 写日志
            if (raw_found) {
                ++detected;
            }
            csv << index << ',' << (raw_found ? 1 : 0) << ','
                << cv::format("%.2f", raw.center.x) << ','
                << cv::format("%.2f", raw.center.y) << ','
                << cv::format("%.2f", raw.width) << ','
                << cv::format("%.2f", raw.height) << ','
                << cv::format("%.2f", raw.angle_degrees) << ','
                << (pose_result.valid ? 1 : 0) << ','
                << cv::format("%.2f", pose_result.position_mm.x) << ','
                << cv::format("%.2f", pose_result.position_mm.y) << ','
                << cv::format("%.2f", pose_result.position_mm.z) << ','
                << cv::format("%.2f", pose_result.distance_mm) << ','
                << cv::format("%.2f", pose_result.euler_deg[0]) << ','
                << cv::format("%.2f", pose_result.euler_deg[1]) << ','
                << cv::format("%.2f", pose_result.euler_deg[2]) << ','
                << cv::format("%.3f", pose_result.reprojection_error_px) << '\n';

            if (index % kReportStep == 0) {
                std::cout << "frame " << index << " 候选=" << debug.raw_candidates
                          << " 检测=" << (raw_found ? "是" : "否") << " 状态="
                          << (shown.found ? (holding ? "预测" : "命中") : "未检测");
                if (pose_result.valid) {
                    std::cout << " 距离=" << cvRound(pose_result.distance_mm) << "mm"
                              << " 误差=" << cv::format("%.2f", pose_result.reprojection_error_px)
                              << "px";
                }
                std::cout << '\n';
            }
            ++index;
        }
        writer.release();
        capture.release();
        csv.close();

        if (index == 0) {
            throw std::runtime_error("视频没有可读取的帧");
        }
        std::cout << input_path.filename().string() << ": 共 " << index << " 帧，"
                  << "单帧直接命中 " << detected << " 帧 (" << std::fixed
                  << std::setprecision(2) << 100.0 * detected / index << "%)\n";
        if (!distances.empty()) {
            const double mean =
                std::accumulate(distances.begin(), distances.end(), 0.0) /
                static_cast<double>(distances.size());
            double variance = 0.0;
            for (double value : distances) {
                variance += (value - mean) * (value - mean);
            }
            variance /= static_cast<double>(distances.size());
            std::cout << "位姿: 成功 " << solved << " 帧，距离 " << mean
                      << " ± " << std::sqrt(variance) << " mm（"
                      << *std::min_element(distances.begin(), distances.end()) << " ~ "
                      << *std::max_element(distances.begin(), distances.end())
                      << " mm）\n";
        }
        std::cout << "结果视频: " << video_path.string() << "\n"
                  << "逐帧日志: " << csv_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}
