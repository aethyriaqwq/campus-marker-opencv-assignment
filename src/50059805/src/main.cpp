#include "Calibrator.h"
#include "PoseEstimator.h"
#include "VideoProcessor.h"
#include "Visualizer.h"

void runVideoProcessor(const std::string& videoPath) {
  VideoProcessor processor;
  processor.run(videoPath);
}

void runChallengePart() {
  std::cout << "Running challenge part: Camera calibration and marker detection"
            << std::endl;
  Calibrator calibrator;
  std::string params_file = "camera_params.yml";
  std::string calib_video_path = "../../../data/raw/calibration_video.avi";
  std::string marker_video_path = "../../../data/raw/marker_video.avi";
  if (calibrator.loadParams(params_file)) {
    std::cout << "Loaded camera parameters from " << params_file << std::endl;
  } else {
    cv::VideoCapture calib_cap(calib_video_path);
    if (!calib_cap.isOpened()) {
      std::cerr << "Failed to open calibration video: " << calib_video_path
                << std::endl;
      return;
    }
    std::cout << "Starting camera calibration using video: " << calib_video_path
              << std::endl;

    std::vector<std::vector<cv::Point2f> > image_points;
    cv::Mat frame;
    int frame_count = 0;

    while (calib_cap.read(frame)) {
      if (frame.empty()) {
        std::cerr << "Empty frame captured from calibration video."
                  << std::endl;
        frame_count++;
        continue;
      }
      if (frame_count % 30 == 0) {
        std::vector<cv::Point2f> corners;
        if (calibrator.extractCorners(frame, corners)) {
          image_points.push_back(corners);
        }
      }
      frame_count++;
    }
    calib_cap.release();
    std::cout << "Extracted corners from " << image_points.size() << " frames."
              << std::endl;

    if (image_points.size() < 10) {
      std::cerr << "Not enough corners detected for calibration." << std::endl;
      return;
    }
    double error = calibrator.calibrate(image_points, cv::Size(7, 7), 0.03f,
                                        cv::Size(1440, 1080));
    calibrator.saveParams(params_file);
  }

  PoseEstimator pose_estimator(calibrator.getCameraMatrix(),
                               calibrator.getDistCoeffs());
  MarkerDetector detector;
  Visualizer visualizer;

  float marker_size = 0.08f;  // Marker 大小（单位 m）
  std::vector<cv::Point3f> marker_3d_points = {
      cv::Point3f(-marker_size / 2, -marker_size / 2, 0),
      cv::Point3f(marker_size / 2, -marker_size / 2, 0),
      cv::Point3f(marker_size / 2, marker_size / 2, 0),
      cv::Point3f(-marker_size / 2, marker_size / 2, 0)};

  cv::VideoCapture marker_cap(marker_video_path);
  if (!marker_cap.isOpened()) {
    std::cerr << "Failed to open marker video: " << marker_video_path
              << std::endl;
    return;
  }

  std::cout << "Starting marker detection and pose estimation using video: "
            << marker_video_path << std::endl;

  cv::Mat frame;

  std::string output_video_path = "output_marker_detection.avi";
  int output_fps = 30;
  int max_processed_frames = 300;  // 最多处理的帧数
  int processed_frames = 0;

  int frame_width = static_cast<int>(marker_cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height =
      static_cast<int>(marker_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  cv::Size frame_size(frame_width, frame_height);

  cv::VideoWriter video_writer(output_video_path,
                               cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                               output_fps, frame_size, true);

  if(!video_writer.isOpened()) {
    std::cerr << "Failed to open video writer for output: " << output_video_path
              << std::endl;
    return;
  }

  int frame_count = 0;

  while (marker_cap.read(frame)) {
    if (frame.empty()) {
      std::cerr << "Empty frame captured from marker video." << std::endl;
      continue;
    }

    frame_count++;
    if (frame_count > max_processed_frames) {
      std::cout << "Reached maximum processed frames: " << max_processed_frames << std::endl;
      break;
    }

    MarkerResult result = detector.detect(frame);
    if (result.detected_ == true && result.points_.size() == 4) {
      if (pose_estimator.solve(result.points_, marker_3d_points)) {
        pose_estimator.drawAxis(frame, 0.04f);
      }
    }
    visualizer.drawMarker(frame, result);

    if (video_writer.isOpened()) {
      video_writer.write(frame);
      processed_frames++;
    }

    cv::imshow("Marker Detection and Pose Estimation", frame);
    if (cv::waitKey(30) == 27) {  // 按 ESC 键退出
      break;
    }
  }
  marker_cap.release();
  if (video_writer.isOpened()) {
    video_writer.release();
    std::cout << "Output video saved to: " << output_video_path << std::endl;
  }
  cv::destroyAllWindows();
}
int main(int argc, char** argv) {
  if (argc < 2) {
    runVideoProcessor("../../../data/raw/marker_video.avi");
    return 0;
  }

  std::string mode = argv[1];
  if (mode == "mandatory") {
    runVideoProcessor("../../../data/raw/marker_video.avi");
  } else if (mode == "challenge") {
    runChallengePart();
  } else {
    std::cerr << "Unknown mode: " << mode << std::endl;
    std::cerr << "Usage: " << argv[0] << " [mandatory|challenge]" << std::endl;
  }
  return 0;
}