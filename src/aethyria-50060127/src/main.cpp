#include "calibrate.hpp"
#include "detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

struct Options {
  std::string input;
  std::string output;
  std::string video;
  std::string camera = "src/aethyria-50060127/camera.yaml";
  std::string model = "src/aethyria-50060127/marker.yaml";
  std::string params = "src/aethyria-50060127/detector.yaml";
  int save_every = 1;
  int from = 0;
  // Negative means the window runs through the last frame. The command line
  // itself does not accept a negative index; this is only the omitted --to.
  int to = -1;
  bool calibrate = false;
  bool output_set = false;
  bool video_set = false;
  bool save_every_set = false;
  bool from_set = false;
  bool to_set = false;
  bool camera_set = false;
  bool model_set = false;
  bool params_set = false;
};

void usage(std::ostream& out) {
  out << "usage: marker_detect --input <video> [--output-dir <dir>] [--save-every <n>]\n"
         "                      [--output-video <file>] [--from <frame>] [--to <frame>]\n"
         "       marker_detect --calibrate --input <video> [--camera <yaml>]\n"
         "       marker_detect --help\n"
         "\n"
         "--output-dir writes frame_NNNN.png. Omit --save-every to write every selected frame.\n"
         "--save-every is a positive interval and requires --output-dir. The first selected frame is written.\n"
         "--output-video writes every selected frame as avc1 at the container frame rate.\n"
         "--from is inclusive, default 0. Earlier frames are detected only to keep the track.\n"
         "--to is inclusive. Omit it to read through the end. Reading stops after this frame.\n"
         "--camera is the calibration yaml. Detection reads it. The default is src/aethyria-50060127/camera.yaml.\n"
         "--model is the marker geometry. --params is the estimator. Tuning either file does not change the program.\n";
}

bool parse(int argc, char** argv, Options& options, std::string& error) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](std::string& out) {
      if (i + 1 >= argc || std::string(argv[i + 1]).empty()) {
        error = "missing value for " + arg;
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
        std::size_t used = 0;
        const int value = std::stoi(text, &used);
        if (used != text.size()) {
          error = "not an integer: " + text;
          return false;
        }
        out = value;
      } catch (const std::exception&) {
        error = "not an integer: " + text;
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
      options.output_set = true;
    } else if (arg == "--save-every") {
      if (!need_int(options.save_every)) {
        return false;
      }
      options.save_every_set = true;
    } else if (arg == "--output-video") {
      if (!need(options.video)) {
        return false;
      }
      options.video_set = true;
    } else if (arg == "--from") {
      if (!need_int(options.from)) {
        return false;
      }
      options.from_set = true;
    } else if (arg == "--to") {
      if (!need_int(options.to)) {
        return false;
      }
      options.to_set = true;
    } else if (arg == "--calibrate") {
      options.calibrate = true;
    } else if (arg == "--camera") {
      if (!need(options.camera)) {
        return false;
      }
      options.camera_set = true;
    } else if (arg == "--model") {
      if (!need(options.model)) {
        return false;
      }
      options.model_set = true;
    } else if (arg == "--params") {
      if (!need(options.params)) {
        return false;
      }
      options.params_set = true;
    } else {
      error = "unknown option " + arg;
      return false;
    }
  }
  if (options.input.empty()) {
    error = "missing --input";
    return false;
  }
  if (options.calibrate) {
    if (options.output_set || options.save_every_set || options.video_set || options.from_set || options.to_set ||
        options.model_set || options.params_set) {
      error = "--calibrate only accepts --input and --camera";
      return false;
    }
    return true;
  }
  if (options.save_every_set && !options.output_set) {
    error = "--save-every requires --output-dir";
    return false;
  }
  if (options.save_every_set && options.save_every < 1) {
    error = "--save-every must be a positive integer";
    return false;
  }
  if (options.from < 0) {
    error = "--from must be a frame index >= 0";
    return false;
  }
  if (options.to_set && options.to < 0) {
    error = "--to must be a frame index >= 0";
    return false;
  }
  if (options.to_set && options.to < options.from) {
    error = "--to is before --from";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage(std::cout);
      return 0;
    }
  }
  Options options;
  std::string error;
  if (!parse(argc, argv, options, error)) {
    if (!error.empty()) {
      std::cerr << error << "\n";
    }
    usage(std::cerr);
    return 2;
  }
  if (options.calibrate) {
    return calibrate_camera(options.input, options.camera) ? 0 : 1;
  }
  cv::FileStorage intrinsics(options.camera, cv::FileStorage::READ);
  if (!intrinsics.isOpened()) {
    std::cerr << "cannot open " << options.camera << "\n";
    return 1;
  }
  cv::Mat camera;
  cv::Mat dist;
  intrinsics["camera_matrix"] >> camera;
  intrinsics["distortion_coefficients"] >> dist;
  if (camera.rows != 3 || camera.cols != 3) {
    std::cerr << "camera_matrix missing in " << options.camera << "\n";
    return 1;
  }
  MarkerDetector detector;
  if (!detector.load(options.model, options.params, camera, dist, error)) {
    std::cerr << error << "\n";
    return 1;
  }
  cv::VideoCapture capture(options.input);
  if (!capture.isOpened()) {
    std::cerr << "cannot open " << options.input << "\n";
    return 1;
  }
  if (!options.output.empty()) {
    std::filesystem::create_directories(options.output);
  }
  if (!options.video.empty()) {
    const std::filesystem::path video_path(options.video);
    if (video_path.has_parent_path()) {
      std::filesystem::create_directories(video_path.parent_path());
    }
  }
  cv::VideoWriter writer;
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
  while (options.to < 0 || index <= options.to) {
    if (!capture.read(frame)) {
      break;
    }
    if (index < options.from) {
      // Keep the carried plate, but do not count or save frames before the window.
      detector.detect(frame);
    } else {
      Detection detection = detector.detect(frame);
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
      if (save || !options.video.empty()) {
        draw_detection(frame, detection);
      }
      if (save) {
        std::ostringstream path;
        path << options.output << "/frame_" << std::setw(4) << std::setfill('0') << index << ".png";
        if (!cv::imwrite(path.str(), frame)) {
          std::cerr << "cannot write " << path.str() << "\n";
          return 1;
        }
      }
      if (!options.video.empty()) {
        if (!writer.isOpened()) {
          const double fps = capture.get(cv::CAP_PROP_FPS);
          if (!(fps > 0.0)) {
            std::cerr << "input has no frame rate\n";
            return 1;
          }
          writer.open(options.video, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps,
                      frame.size());
          if (!writer.isOpened()) {
            std::cerr << "cannot open " << options.video << "\n";
            return 1;
          }
        }
        writer.write(frame);
      }
    }
    ++index;
  }
  if (run_start >= 0) {
    close_run(index - 1);
  }
  if (frames == 0) {
    std::cerr << "no frames in range\n";
    return 1;
  }
  std::cout << "frames " << frames << "\n"
            << "detected " << found << "\n"
            << "measured4 " << by_support[4] << "\n"
            << "measured3 " << by_support[3] << "\n"
            << "measured2 " << by_support[2] << "\n"
            << "measured1 " << by_support[1] << "\n"
            << "undetected " << (frames - found) << "\n"
            << "longest_miss " << max_miss << "\n";
  return 0;
}
