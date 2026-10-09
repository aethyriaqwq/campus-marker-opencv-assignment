#pragma once

#include "marker_model.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <string>

struct Detection {
  bool found = false;
  // Outer corners of the plate, drawing order LT, RT, RB, LB.
  // Published together when the luminous pattern matches in exactly one
  // orientation and the outer edges lie on the bright boundary.
  // Otherwise the frame is undetected.
  std::array<cv::Point2f, 4> corners{};
  std::array<bool, 4> measured{};
  int support = 0;
  // Median bright-sample contrast minus median dark-sample contrast.
  float score = 0.f;
  // Model corner that carries the notch. -1 when this frame has no pose.
  int notch = -1;
};

class MarkerDetector {
 public:
  bool load(const std::string& model_path, const std::string& params_path, const cv::Mat& camera,
            const cv::Mat& dist, std::string& error);
  Detection detect(const cv::Mat& bgr);
  const MarkerModel& model() const { return model_; }

 private:
  struct Pose {
    cv::Vec3d r{0, 0, 0};
    cv::Vec3d t{0, 0, 1};
    bool valid = false;
  };
  struct Orientation {
    int confirmed[4] = {};
    int contradicted[4] = {};
    bool feasible[4] = {};
    // Notch-gap runs that have enough on-screen samples, and how many of
    // those runs agreed. A gap that is off screen is not present, so it
    // cannot make a pose unique by hiding the station that fixes the scale.
    int gap_present[4] = {};
    int gap_confirmed[4] = {};
    double separation = 0;
    // Unique means turn 0 was read, the other three were not, and the notch
    // station itself was in the image. An off-screen gap is not a reading.
    bool unique() const {
      return feasible[0] && !feasible[1] && !feasible[2] && !feasible[3] && gap_confirmed[0] >= 1 &&
             gap_confirmed[0] == gap_present[0];
    }
  };
  struct EdgePoint {
    cv::Point2d p;
    cv::Point2d n;
    // 0 top, 1 right, 2 bottom, 3 left. A side is one line correspondence.
    int side = 0;
  };
  // Edge samples whose contrast peak is within the Huber band are locked.
  // A code match inside a wide arm is not yet a border.
  struct EdgeFit {
    std::vector<int> index;
    std::vector<double> residual;
    int visible = 0;
    double cost = 0;
    bool enough = false;
    bool locked = false;
    // Three sides each spanned by at least half the outer side. A track that
    // already sits on the border may keep it as a side slides out of frame.
    // A new pose may not: a short segment does not determine the scale.
    bool determined = false;
  };

  MarkerModel model_;
  DetectorParams params_;
  cv::Mat camera_;
  cv::Mat dist_;
  std::vector<EdgePoint> edges_;
  std::vector<int> interior_;
  Pose pose_;
  // Per-frame pose step. The search reaches one stroke. A seed that stays
  // put loses the border when the plate moves farther than that. The step is
  // kept only after a pose is accepted.
  cv::Vec3d vel_r_{};
  cv::Vec3d vel_t_{};
  // Set when a locked border has matched the notch code once. Later frames
  // keep that identity until the code contradicts it. A weak read does not
  // throw the border away.
  bool identity_ = false;
  bool ready_ = false;

  std::vector<cv::Point2d> project(const Pose& pose, const std::vector<cv::Point3d>& object) const;
  Pose predict() const;
  bool within_basin(const Pose& seed, const Pose& refined) const;
  bool code_accepts(const Orientation& code, bool acquiring) const;
  void remember(const Pose& next, bool adopt_velocity);
  void clear_track();
  bool facing(const Pose& pose) const;
  std::array<cv::Point2d, 4> plate(const Pose& pose) const;
  // notch_block rejects a solid run that hides a notch-sized gap. That test
  // needs the pose already on the border; a coarse pose uses majority only.
  Orientation orient(const Pose& pose, const cv::Mat& gray, bool notch_block) const;
  double edge_offset(const Pose& pose, const EdgePoint& edge, const cv::Mat& gray, bool& on_screen) const;
  EdgeFit fit_edges(const Pose& pose, const cv::Mat& gray) const;
  Pose refine(const Pose& pose, const cv::Mat& gray) const;
  bool cold_start(const cv::Mat& gray, Pose& out) const;
  Detection publish(const Pose& pose, double score) const;
};

void draw_detection(cv::Mat& bgr, const Detection& detection);
