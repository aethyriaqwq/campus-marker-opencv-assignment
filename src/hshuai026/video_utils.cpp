#include "video_utils.hpp"

#include <cctype>
#include <iostream>
#include <vector>

namespace video {

namespace fs = std::filesystem;

// 定位输入素材
fs::path resolveInputPath(const fs::path& given) {
    std::error_code error;
    if (fs::is_regular_file(given, error)) {
        return given;
    }

    // 从当前目录与可执行文件目录分别向上搜索 data/raw/
    const std::string name = given.filename().string();
    std::vector<fs::path> starts;
    starts.push_back(fs::current_path(error));
    const fs::path self = fs::read_symlink("/proc/self/exe", error);
    if (!error && !self.empty()) {
        starts.push_back(self.parent_path());
    }

    for (const fs::path& start : starts) {
        fs::path dir = start;
        for (int depth = 0; depth < 6 && !dir.empty(); ++depth) {
            const fs::path candidate = dir / "data" / "raw" / name;
            if (fs::is_regular_file(candidate, error)) {
                std::cout << "提示: 已自动定位素材 " << candidate.string() << "\n";
                return candidate;
            }
            if (!dir.has_parent_path() || dir.parent_path() == dir) {
                break;
            }
            dir = dir.parent_path();
        }
    }
    return given;
}

// 解析输出目录
fs::path resolveOutputDir(const fs::path& given, const fs::path& input) {
    std::error_code error;
    if (given.is_absolute()) {
        return given;
    }
    const fs::path root = input.parent_path().parent_path().parent_path();
    if (!root.empty() && fs::is_directory(root / "data", error)) {
        return root / given;
    }
    return given;
}

// 打开视频源
bool openCapture(cv::VideoCapture& capture, const std::string& input) {
    const bool camera_input =
        input.size() == 1 && std::isdigit(static_cast<unsigned char>(input[0])) != 0;
    if (camera_input) {
        capture.open(std::stoi(input));
    } else {
        capture.open(input);
    }
    return capture.isOpened();
}

// 打开视频写出器
bool openWriter(cv::VideoWriter& writer, const fs::path& path, double fps,
                const cv::Size& size) {
    std::error_code remove_error;
    fs::remove(path, remove_error);  // 删除旧文件，避免残留数据
    auto try_open = [&](const char* fourcc) {
        writer.open(path.string(),
                    cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]),
                    fps, size, true);
        return writer.isOpened();
    };
    if (try_open("avc1")) {
        return true;
    }
    if (try_open("H264")) {
        return true;
    }
    std::cout << "提示: 本机 OpenCV 无 H.264 编码器，回退 MPEG-4 (mp4v)\n";
    return try_open("mp4v");
}

}  // namespace video
