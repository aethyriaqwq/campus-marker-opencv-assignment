#pragma once

#include <string>

#include "MarkerDetector.h"

/*
 * VideoProcessor 类用于处理视频流，检测标记，并根据检测结果更新状态
 * 它使用 MarkerDetector
 * 类来检测每一帧中的标记，并根据连续的检测结果来判断标记是否丢失或重新检测到
 * 当标记连续丢失超过 MAX_LOST_FRAMES 帧时，状态将被重置
 * 当标记连续检测到超过 MIN_DETECTED_FRAMES 帧时，状态将被更新为检测到标记
 */

class VideoProcessor {
 public:
  VideoProcessor();
  void run(const std::string& videoPath);

 private:
  MarkerDetector detector_;
  int lostFrames_ = 0;
  int detectedFrames_ = 0;
  const int MAX_LOST_FRAMES = 5;
  const int MIN_DETECTED_FRAMES = 3;
  MarkerResult lastResult_;

  void updateState(const MarkerResult& currentResult);
};