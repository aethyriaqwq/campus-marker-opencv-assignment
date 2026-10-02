// 相机标定：检测 7x7 圆点阵列标定板，标定内参与畸变系数并输出 YAML。
// 用法: calib_dots <视频> <输出yaml> [点间距mm] [采样间隔] [最少视角数]
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace {

cv::Ptr<cv::SimpleBlobDetector> makeBlobDetector() {
    cv::SimpleBlobDetector::Params params;
    params.filterByColor = true;
    params.blobColor = 0;  // 深色圆点
    params.filterByArea = true;
    params.minArea = 80.0F;
    params.maxArea = 40000.0F;
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity = false;
    params.minThreshold = 5;
    params.maxThreshold = 200;
    params.thresholdStep = 10;
    params.minDistBetweenBlobs = 8.0F;
    return cv::SimpleBlobDetector::create(params);
}

bool detectGrid(const cv::Mat& gray, cv::Size pattern,
                const cv::Ptr<cv::SimpleBlobDetector>& detector,
                std::vector<cv::Point2f>& centers) {
    centers.clear();
    if (!cv::findCirclesGrid(gray, pattern, centers, cv::CALIB_CB_SYMMETRIC_GRID,
                             detector)) {
        return false;
    }
    // 用亚像素角点细化每个圆点中心（圆点中心本身接近角点响应极值）。
    cv::cornerSubPix(gray, centers, cv::Size(7, 7), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                      30, 0.01));
    return true;
}

double meanShift(const std::vector<cv::Point2f>& a,
                 const std::vector<cv::Point2f>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 1e9;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += cv::norm(a[i] - b[i]);
    }
    return sum / static_cast<double>(a.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "用法: calib_dots <视频> <输出yaml> [点间距mm] [采样间隔] "
                     "[最少视角数]\n";
        return 1;
    }
    const fs::path video_path = argv[1];
    const fs::path yaml_path = argv[2];
    const double spacing_mm = argc > 3 ? std::atof(argv[3]) : 1.0;
    const int step = argc > 4 ? std::max(1, std::atoi(argv[4])) : 20;
    const int min_views = argc > 5 ? std::atoi(argv[5]) : 12;
    constexpr int kCols = 7;
    constexpr int kRows = 7;

    cv::VideoCapture capture(video_path.string());
    if (!capture.isOpened()) {
        std::cerr << "打不开视频: " << video_path << "\n";
        return 1;
    }
    const cv::Size image_size(
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cv::Ptr<cv::SimpleBlobDetector> detector = makeBlobDetector();

    std::vector<cv::Point3f> model;
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            model.emplace_back(static_cast<float>(col * spacing_mm),
                               static_cast<float>(row * spacing_mm), 0.0F);
        }
    }

    struct View {
        std::vector<cv::Point2f> centers;
        int frame = 0;
        double sharpness = 0.0;
        double span = 0.0;
    };
    std::vector<View> views;
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<int> used_frames;
    std::vector<std::vector<cv::Point2f>> saved_centers;
    std::vector<int> saved_frames;
    std::vector<cv::Point2f> last;
    cv::Mat frame;
    cv::Mat gray;
    int index = 0;
    int detections = 0;
    while (capture.read(frame)) {
        if (index % step == 0) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Point2f> centers;
            if (detectGrid(gray, cv::Size(kCols, kRows), detector, centers)) {
                ++detections;
                // 清晰度（Laplacian 标准差）与点阵跨度：模糊或太小的画面精度差。
                cv::Mat laplacian;
                cv::Laplacian(gray, laplacian, CV_64F);
                cv::Scalar mean_lap;
                cv::Scalar std_lap;
                cv::meanStdDev(laplacian, mean_lap, std_lap);
                const double sharpness = std_lap[0];
                const double span = cv::norm(centers.front() - centers.back());
                views.push_back({centers, index, sharpness, span});
            }
        }
        ++index;
    }
    std::cout << "扫描 " << index << " 帧（每 " << step << " 帧取一张），检出点阵 "
              << detections << " 次\n";
    if (views.empty()) {
        std::cerr << "没有检出任何点阵\n";
        return 1;
    }
    // 按"清晰度 × 点阵跨度"排序，跨度和清晰度都在中位数以上的优先，
    // 再按位移去重，最后均匀抽稀到不超过 45 个视角。
    std::vector<double> sharp_values;
    std::vector<double> span_values;
    for (const View& view : views) {
        sharp_values.push_back(view.sharpness);
        span_values.push_back(view.span);
    }
    auto percentile = [](std::vector<double> values, double q) {
        std::sort(values.begin(), values.end());
        const std::size_t index = static_cast<std::size_t>(
            std::min<double>(values.size() - 1, q * (values.size() - 1)));
        return values[index];
    };
    const double sharp_min = percentile(sharp_values, 0.35);
    const double span_min = percentile(span_values, 0.35);
    std::cout << "筛选阈值：清晰度 >= " << cvRound(sharp_min) << "，跨度 >= "
              << cvRound(span_min) << " px\n";
    for (const View& view : views) {
        if (view.sharpness < sharp_min || view.span < span_min) {
            continue;
        }
        if (!last.empty() && meanShift(last, view.centers) <= 25.0) {
            continue;
        }
        object_points.push_back(model);
        image_points.push_back(view.centers);
        used_frames.push_back(view.frame);
        last = view.centers;
        if (saved_centers.size() < 4) {
            saved_centers.push_back(view.centers);
            saved_frames.push_back(view.frame);
        }
    }
    std::cout << "筛选后视角数: " << image_points.size() << "\n";
    constexpr std::size_t kMaxViews = 45;
    if (image_points.size() > kMaxViews) {
        std::vector<std::vector<cv::Point3f>> reduced_objects;
        std::vector<std::vector<cv::Point2f>> reduced_images;
        std::vector<int> reduced_frames;
        const double ratio =
            static_cast<double>(image_points.size()) / static_cast<double>(kMaxViews);
        for (std::size_t i = 0; i < kMaxViews; ++i) {
            const std::size_t source =
                std::min(image_points.size() - 1, static_cast<std::size_t>(i * ratio));
            reduced_objects.push_back(object_points[source]);
            reduced_images.push_back(image_points[source]);
            reduced_frames.push_back(used_frames[source]);
        }
        object_points.swap(reduced_objects);
        image_points.swap(reduced_images);
        used_frames.swap(reduced_frames);
        std::cout << "抽稀后视角数: " << image_points.size() << "\n";
    }
    if (static_cast<int>(image_points.size()) < min_views) {
        std::cerr << "有效视角不足（需要至少 " << min_views << " 个）\n";
        return 1;
    }

    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    // 用固定 k3 的 4 参数畸变模型，避免高次项过拟合；再用每视角误差剔除异常视角。
    std::vector<double> per_view_errors;
    cv::Mat std_dev_intrinsics;
    cv::Mat std_dev_extrinsics;
    double rms = cv::calibrateCamera(object_points, image_points, image_size,
                                     camera_matrix, dist_coeffs, rvecs, tvecs,
                                     cv::CALIB_FIX_K3);
    rms = cv::calibrateCamera(object_points, image_points, image_size, camera_matrix,
                              dist_coeffs, rvecs, tvecs, cv::CALIB_FIX_K3,
                              cv::TermCriteria(cv::TermCriteria::COUNT +
                                                   cv::TermCriteria::EPS,
                                               50, 1e-6));
    // C++ 里带 perViewErrors 的重载还需要两个标准差输出（Python 里叫
    // calibrateCameraExtended）。
    rms = cv::calibrateCamera(object_points, image_points, image_size, camera_matrix,
                              dist_coeffs, rvecs, tvecs, std_dev_intrinsics,
                              std_dev_extrinsics, per_view_errors, cv::CALIB_FIX_K3);
    // 剔除重投影误差过大的视角后重标一次。
    {
        std::vector<std::vector<cv::Point3f>> kept_objects;
        std::vector<std::vector<cv::Point2f>> kept_images;
        std::vector<int> kept_frames;
        const double limit = std::max(3.0, rms * 2.0);
        for (std::size_t i = 0; i < image_points.size(); ++i) {
            if (per_view_errors[i] <= limit) {
                kept_objects.push_back(object_points[i]);
                kept_images.push_back(image_points[i]);
                kept_frames.push_back(used_frames[i]);
            }
        }
        if (kept_images.size() >= 8 &&
            kept_images.size() < image_points.size()) {
            std::cout << "剔除 " << (image_points.size() - kept_images.size())
                      << " 个误差 > " << limit << " px 的视角\n";
            object_points.swap(kept_objects);
            image_points.swap(kept_images);
            used_frames.swap(kept_frames);
            rms = cv::calibrateCamera(object_points, image_points, image_size,
                                      camera_matrix, dist_coeffs, rvecs, tvecs,
                                      std_dev_intrinsics, std_dev_extrinsics,
                                      per_view_errors, cv::CALIB_FIX_K3);
        }
    }
    double worst = 0.0;
    double sum = 0.0;
    for (double error : per_view_errors) {
        worst = std::max(worst, error);
        sum += error;
    }
    std::cout << std::fixed << std::setprecision(4)
              << "标定完成：视角数 " << image_points.size()
              << "，重投影误差 RMS = " << rms << " px，平均 "
              << (sum / std::max<std::size_t>(1, per_view_errors.size()))
              << " px，最差 " << worst << " px\n"
              << "内参矩阵:\n" << camera_matrix << "\n畸变系数:\n"
              << dist_coeffs << "\n";

    cv::FileStorage storage(yaml_path.string(), cv::FileStorage::WRITE);
    storage << "board_type" << "7x7 symmetric dot grid (black frame)";
    storage << "dot_spacing_mm" << spacing_mm;
    storage << "image_width" << image_size.width;
    storage << "image_height" << image_size.height;
    storage << "rms_reprojection_error_px" << rms;
    storage << "views_used" << static_cast<int>(image_points.size());
    storage << "camera_matrix" << camera_matrix;
    storage << "distortion_coefficients" << dist_coeffs;
    storage << "used_frames" << used_frames;
    storage.release();
    std::cout << "已保存: " << yaml_path.string() << "\n";

    const fs::path evidence = yaml_path.parent_path();
    for (std::size_t i = 0; i < saved_centers.size(); ++i) {
        cv::Mat sample;
        capture.set(cv::CAP_PROP_POS_FRAMES, saved_frames[i]);
        if (!capture.read(sample) || sample.empty()) {
            continue;
        }
        cv::drawChessboardCorners(sample, cv::Size(kCols, kRows), saved_centers[i],
                                  true);
        std::ostringstream name;
        name << "calib_dots_" << i << ".jpg";
        cv::Mat half;
        cv::resize(sample, half, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        cv::imwrite((evidence / name.str()).string(), half,
                    {cv::IMWRITE_JPEG_QUALITY, 90});
    }
    std::cout << "已输出 " << saved_centers.size() << " 张点阵检出示意图到 "
              << evidence << "\n";
    return 0;
}
