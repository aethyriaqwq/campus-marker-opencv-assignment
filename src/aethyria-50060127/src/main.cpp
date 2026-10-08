#include "calibrate.hpp"
#include "detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

struct Options {
  std::string input;
  std::string output;
  std::string camera = "src/aethyria-50060127/camera.yaml";
  int save_every = 0;
  int from = 0;
  int to = -1;
  bool calibrate = false;
};

void usage() {
  std::cerr << "usage: marker_detect --input <video> [--output-dir <dir>] [--save-every <n>]\n"
               "                      [--from <frame>] [--to <frame>]\n"
               "       marker_detect --calibrate --input <video> [--camera <yaml>]\n"
               "example:\n"
               "  marker_detect --input data/raw/marker_video.avi \\\n"
               "      --output-dir output/aethyria-50060127/frames --save-every 20\n";
}

bool parse(int argc, char** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](std::string& out) {
      if (i + 1 >= argc) {
        return false;
      }
      out = argv[++i];
      return true;
    };
    auto need_int = [&](int& out) {
      std::string text;
      if (!need(text)) {
        return false;
      }
      try {
        out = std::stoi(text);
      } catch (const std::exception&) {
        return false;
      }
      return true;
    };
    if (arg == "--input") {
      if (!need(options.input)) {
        return false;
      }
    } else if (arg == "--output-dir") {
      if (!need(options.output)) {
        return false;
      }
    } else if (arg == "--save-every") {
      if (!need_int(options.save_every)) {
        return false;
      }
    } else if (arg == "--from") {
      if (!need_int(options.from)) {
        return false;
      }
    } else if (arg == "--to") {
      if (!need_int(options.to)) {
        return false;
      }
    } else if (arg == "--calibrate") {
      options.calibrate = true;
    } else if (arg == "--camera") {
      if (!need(options.camera)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return !options.input.empty() && options.save_every >= 0 && options.from >= 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse(argc, argv, options)) {
    usage();
    return 2;
  }
  if (options.calibrate) {
    return calibrate_camera(options.input, options.camera) ? 0 : 1;
  }
  cv::VideoCapture capture(options.input);
  if (!capture.isOpened()) {
    std::cerr << "cannot open " << options.input << "\n";
    return 1;
  }
  if (!options.output.empty()) {
    std::filesystem::create_directories(options.output);
  }
  int frames = 0;
  int found = 0;
  int by_support[5] = {};
  int miss_run = 0;
  int max_miss = 0;
  int index = 0;
  int run_start = -1;
  int run_support = -1;
  cv::Mat frame;
  auto close_run = [&](int end) {
    if (run_start < 0) {
      return;
    }
    std::cout << (run_support == 0 ? "undetected " : "detected ") << run_start << "-" << end;
    if (run_support > 0) {
      std::cout << " support " << run_support;
    }
    std::cout << "\n";
  };
  while (capture.read(frame)) {
    const bool selected = index >= options.from && (options.to < 0 || index <= options.to);
    if (index < options.from) {
      // Keep the carried plate, but do not count or save frames before the window.
      detect_marker(frame);
    } else if (!selected) {
      if (run_start >= 0) {
        close_run(index - 1);
        run_start = -1;
      }
    } else {
      Detection detection = detect_marker(frame);
      ++frames;
      const int support = detection.found ? detection.support : 0;
      if (run_start < 0) {
        run_start = index;
        run_support = support;
      } else if (support != run_support) {
        close_run(index - 1);
        run_start = index;
        run_support = support;
      }
      if (detection.found) {
        ++found;
        if (detection.support >= 0 && detection.support <= 4) {
          ++by_support[detection.support];
        }
        miss_run = 0;
      } else {
        ++miss_run;
        max_miss = std::max(max_miss, miss_run);
      }
      const bool save = !options.output.empty() && options.save_every > 0 &&
                        ((index - options.from) % options.save_every == 0);
      if (save) {
        draw_detection(frame, detection);
        std::ostringstream path;
        path << options.output << "/frame_" << std::setw(4) << std::setfill('0') << index << ".png";
        if (!cv::imwrite(path.str(), frame)) {
          std::cerr << "cannot write " << path.str() << "\n";
          return 1;
        }
      }
    }
    ++index;
  }
  if (run_start >= 0) {
    close_run(index - 1);
  }
  std::cout << "frames " << frames << "\n"
            << "detected " << found << "\n"
            << "homography " << by_support[4] << "\n"
            << "affine " << by_support[3] << "\n"
            << "similarity " << by_support[2] << "\n"
            << "single " << by_support[1] << "\n"
            << "undetected " << (frames - found) << "\n"
            << "longest_miss " << max_miss << "\n";
  return frames == 0 ? 1 : 0;
}
