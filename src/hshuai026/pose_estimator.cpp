#include "pose_estimator.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>

namespace pose {

// 读取标定结果
bool CameraIntrinsics::load(const std::string& yaml_path) {
    cv::FileStorage storage(yaml_path, cv::FileStorage::READ);
    if (!storage.isOpened()) {
        return false;
    }
    storage["camera_matrix"] >> camera_matrix;
    storage["distortion_coefficients"] >> dist_coeffs;
    storage.release();
    return !camera_matrix.empty();
}

// 生成标志物模型点
std::vector<cv::Point3f> markerModel(float size_mm) {
    const float half = size_mm / 2.0F;
    return {{-half, -half, 0.0F},
            {half, -half, 0.0F},
            {half, half, 0.0F},
            {-half, half, 0.0F}};
}

// 求解位姿：solvePnP + 投影回图像计算重投影误差
Result solve(const std::array<cv::Point2f, 4>& corners, const CameraIntrinsics& camera,
             float size_mm) {
    Result result;
    const std::vector<cv::Point3f> model = markerModel(size_mm);
    const std::vector<cv::Point2f> image_points(corners.begin(), corners.end());

    cv::Mat rvec;
    cv::Mat tvec;
    if (!cv::solvePnP(model, image_points, camera.camera_matrix, camera.dist_coeffs, rvec,
                      tvec, false, cv::SOLVEPNP_IPPE)) {
        return result;
    }

    // 重投影误差
    std::vector<cv::Point2f> projected;
    cv::projectPoints(model, rvec, tvec, camera.camera_matrix, camera.dist_coeffs,
                      projected);
    double error = 0.0;
    for (std::size_t i = 0; i < projected.size(); ++i) {
        error += cv::norm(projected[i] - image_points[i]);
    }
    result.reprojection_error_px = error / static_cast<double>(projected.size());

    // 位置与距离
    result.rvec = rvec.clone();
    result.tvec = tvec.clone();
    result.position_mm = cv::Point3d(tvec.at<double>(0), tvec.at<double>(1),
                                     tvec.at<double>(2));
    result.distance_mm = cv::norm(tvec);

    // 姿态角（ZYX 欧拉角）
    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    const double sy = std::sqrt(rotation.at<double>(0, 0) * rotation.at<double>(0, 0) +
                                rotation.at<double>(1, 0) * rotation.at<double>(1, 0));
    result.euler_deg = {
        std::atan2(rotation.at<double>(2, 1), rotation.at<double>(2, 2)) * 180.0 / CV_PI,
        std::atan2(-rotation.at<double>(2, 0), sy) * 180.0 / CV_PI,
        std::atan2(rotation.at<double>(1, 0), rotation.at<double>(0, 0)) * 180.0 /
            CV_PI};
    result.valid = true;
    return result;
}

}  // namespace pose
