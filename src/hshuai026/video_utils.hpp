// 视频与路径工具：素材定位、摄像头输入、视频写出。
#pragma once

#include <opencv2/videoio.hpp>

#include <filesystem>
#include <string>

namespace video {

// 定位输入素材（支持相对路径、自动向上搜索 data/raw/）
std::filesystem::path resolveInputPath(const std::filesystem::path& given);

// 解析输出目录（相对路径以仓库根为基准）
std::filesystem::path resolveOutputDir(const std::filesystem::path& given,
                                       const std::filesystem::path& input);

// 打开视频；纯数字输入按摄像头序号处理
bool openCapture(cv::VideoCapture& capture, const std::string& input);

// 打开写出器（avc1 → H264 → mp4v 依次尝试，写前删除同名旧文件）
bool openWriter(cv::VideoWriter& writer, const std::filesystem::path& path,
                double fps, const cv::Size& size);

}  // namespace video
