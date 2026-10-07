#include "VideoProcessor.h"
#include "Visualizer.h"
#include <iostream>

VideoProcessor::VideoProcessor() {}

// 处理视频流，检测标记，并根据检测结果更新状态
void VideoProcessor::run(const std::string& videoPath) {
  cv::VideoCapture cap(videoPath);
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
    return;
  }
  cv::Mat frame;

  Visualizer visualizer;

  while (cap.read(frame)) {
    if (frame.empty()) {
      std::cerr << "Error: Empty frame captured." << std::endl;
      break;
    }

    MarkerResult currentResult = detector_.detect(frame);
    updateState(currentResult);

    // 根据状态绘制标记检测结果
    if (detectedFrames_ >= MIN_DETECTED_FRAMES && lostFrames_ == 0) {
      visualizer.drawMarker(frame, lastResult_);
    } else {
      MarkerResult emptyResult;
      emptyResult.detected_ = false;
      visualizer.drawMarker(frame, emptyResult);
    }
    cv::imshow("Marker Detection", frame);
    if (cv::waitKey(30) == 27) {  // 按下 'Esc' 键退出
      break;
    }
  }
  cap.release();
  cv::destroyAllWindows();
}

// 更新状态：根据当前检测结果更新连续检测和丢失帧计数
void VideoProcessor::updateState(const MarkerResult& currentResult) {
  if (currentResult.detected_) {
    detectedFrames_++;
    lostFrames_ = 0;
    if (detectedFrames_ >= MIN_DETECTED_FRAMES) {
      lastResult_ = currentResult;
    }
  } else {
    lostFrames_++;
    detectedFrames_ = 0;
    if (lostFrames_ > MAX_LOST_FRAMES) {
      lastResult_.detected_ = false;  // 标记为未检测到
    }
    return;
  }
}