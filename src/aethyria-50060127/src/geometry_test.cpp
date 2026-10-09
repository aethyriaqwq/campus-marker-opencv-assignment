#include "detector.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_fails = 0;

void expect(bool ok, const char* name) {
  if (!ok) {
    std::cout << "FAIL " << name << "\n";
    ++g_fails;
  }
}

void expect_eq(const std::string& got, const std::string& want, const char* name) {
  if (got != want) {
    std::cout << "FAIL " << name << " got " << got << " want " << want << "\n";
    ++g_fails;
  }
}

std::array<cv::Point2d, 4> project_plate(const MarkerModel& model, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                                        const cv::Mat& camera, const cv::Mat& dist) {
  std::vector<cv::Point3d> object;
  for (const cv::Point2d& corner : model.corners) {
    object.push_back({corner.x, corner.y, 0});
  }
  std::vector<cv::Point2d> image;
  cv::projectPoints(object, rvec, tvec, camera, dist, image);
  std::array<cv::Point2d, 4> corners{};
  for (int i = 0; i < 4; ++i) {
    corners[static_cast<std::size_t>(i)] = image[static_cast<std::size_t>(i)];
  }
  return corners;
}

void paint(cv::Mat& image, const std::vector<std::array<cv::Point2d, 4>>& quads, const cv::Vec3d& rvec,
           const cv::Vec3d& tvec, const cv::Mat& camera, const cv::Mat& dist, const cv::Scalar& color) {
  for (const auto& quad : quads) {
    std::vector<cv::Point3d> object;
    for (const cv::Point2d& point : quad) {
      object.push_back({point.x, point.y, 0});
    }
    std::vector<cv::Point2d> projected;
    cv::projectPoints(object, rvec, tvec, camera, dist, projected);
    std::vector<cv::Point> polygon;
    bool ok = true;
    for (const cv::Point2d& point : projected) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
        ok = false;
        break;
      }
      polygon.emplace_back(cvRound(point.x), cvRound(point.y));
    }
    if (ok && polygon.size() == 4) {
      cv::fillConvexPoly(image, polygon, color, cv::LINE_8);
    }
  }
}

cv::Mat render(const MarkerModel& model, const cv::Vec3d& rvec, const cv::Vec3d& tvec, const cv::Mat& camera,
               const cv::Mat& dist, cv::Size size, bool fill_gaps) {
  cv::Mat image(size, CV_8UC3, cv::Scalar(0, 0, 0));
  std::vector<std::array<cv::Point2d, 4>> lamps;
  for (const Lamp& lamp : model.lamps) {
    lamps.push_back(lamp.quad);
  }
  paint(image, lamps, rvec, tvec, camera, dist, {255, 255, 255});
  if (fill_gaps) {
    std::vector<std::array<cv::Point2d, 4>> gaps;
    for (const Lamp& gap : model.gaps) {
      gaps.push_back(gap.quad);
    }
    paint(image, gaps, rvec, tvec, camera, dist, {255, 255, 255});
  }
  cv::GaussianBlur(image, image, {3, 3}, 0.6);
  return image;
}

bool near_quad(const Detection& detection, const std::array<cv::Point2d, 4>& truth, const char* name) {
  if (!detection.found || detection.support != 4) {
    std::cout << "FAIL " << name << " found " << detection.found << " support " << detection.support << "\n";
    ++g_fails;
    return false;
  }
  double side = 0;
  for (int i = 0; i < 4; ++i) {
    side += cv::norm(truth[static_cast<std::size_t>((i + 1) % 4)] - truth[static_cast<std::size_t>(i)]);
  }
  side /= 4.0;
  const double tolerance = std::max(2.0, 0.02 * side);
  bool ok = true;
  for (int i = 0; i < 4; ++i) {
    const cv::Point2d got(detection.corners[static_cast<std::size_t>(i)].x,
                          detection.corners[static_cast<std::size_t>(i)].y);
    const double error = cv::norm(got - truth[static_cast<std::size_t>(i)]);
    if (!detection.measured[static_cast<std::size_t>(i)] || error > tolerance) {
      std::cout << "FAIL " << name << " c" << i << " err " << error << " tol " << tolerance << " got " << got
                << " truth " << truth[static_cast<std::size_t>(i)] << "\n";
      ok = false;
    }
  }
  if (!ok) {
    ++g_fails;
  }
  return ok;
}

}  // namespace

int main() {
  const std::string model_path = "src/aethyria-50060127/marker.yaml";
  const std::string params_path = "src/aethyria-50060127/detector.yaml";
  MarkerModel model;
  std::string error;
  if (!load_model(model_path, model, error)) {
    std::cout << error << "\n";
    return 1;
  }
  expect_eq(model.edge_runs(0), "B30D20B8D8B14", "top edge");
  expect_eq(model.edge_runs(1), "B14D8B8D20B30", "right edge");
  expect_eq(model.edge_runs(2), "B30D20B30", "bottom edge");
  expect_eq(model.edge_runs(3), "B30D20B30", "left edge");
  expect(model.frame_lit(model.arm - 0.5, model.frame * 0.5), "solid arm reaches the common length");
  expect(!model.frame_lit(model.arm + 0.5, model.frame * 0.5), "gap after a solid arm");
  expect(model.frame_lit(model.outer - (model.arm - 0.5), model.frame * 0.5), "notched arm reaches the same length");
  expect(!model.frame_lit(model.outer - (model.arm + 0.5), model.frame * 0.5), "gap before the notched arm");
  const double notch_mid = model.notch_bright + 0.5 * model.notch_gap;
  expect(!model.frame_lit(model.outer - notch_mid, model.frame * 0.5), "top notch is dark");
  expect(model.frame_lit(model.outer - 0.5 * model.notch_bright, model.frame * 0.5), "outer piece of the notched arm");
  expect(!model.frame_lit(model.outer - 0.5 * model.frame, notch_mid), "side notch is dark");
  expect(model.gaps.size() == 2, "notch is two gaps on one L");

  cv::Mat camera = (cv::Mat_<double>(3, 3) << 900, 0, 500, 0, 900, 400, 0, 0, 1);
  cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);
  MarkerDetector detector;
  if (!detector.load(model_path, params_path, camera, dist, error)) {
    std::cout << error << "\n";
    return 1;
  }
  const cv::Size size(1000, 800);
  auto check = [&](MarkerDetector& instance, const char* name, const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                   bool fill_gaps) {
    cv::Mat image = render(model, rvec, tvec, camera, dist, size, fill_gaps);
    const Detection detection = instance.detect(image);
    if (fill_gaps) {
      expect(!detection.found, name);
      return;
    }
    near_quad(detection, project_plate(model, rvec, tvec, camera, dist), name);
  };

  const cv::Vec3d frontal_r(0, 0, 0);
  const cv::Vec3d frontal_t(-40, -40, 360);
  check(detector, "frontal", frontal_r, frontal_t, false);
  double stationary = 0;
  std::array<cv::Point2f, 4> previous{};
  bool have_previous = false;
  for (int i = 0; i < 4; ++i) {
    const Detection detection = detector.detect(render(model, frontal_r, frontal_t, camera, dist, size, false));
    if (!have_previous) {
      previous = detection.corners;
      have_previous = detection.found;
      continue;
    }
    for (int k = 0; k < 4; ++k) {
      stationary = std::max(stationary, cv::norm(detection.corners[static_cast<std::size_t>(k)] -
                                                 previous[static_cast<std::size_t>(k)]));
    }
    previous = detection.corners;
  }
  expect(stationary < 1.0, "stationary corners stay put");
  if (!(stationary < 1.0)) {
    std::cout << "  motion " << stationary << "\n";
  }

  check(detector, "shifted", frontal_r, cv::Vec3d(-25, -40, 360), false);
  const Detection blank = detector.detect(cv::Mat(size, CV_8UC3, cv::Scalar(0, 0, 0)));
  expect(!blank.found, "blank frame clears the plate");
  check(detector, "frontal again", frontal_r, frontal_t, false);
  check(detector, "yaw", cv::Vec3d(0, 0.6, 0), cv::Vec3d(-40, -40, 520), false);
  check(detector, "roll 45", cv::Vec3d(0, 0, 0.8), frontal_t, false);
  check(detector, "roll 90", cv::Vec3d(0, 0, 1.5707963), frontal_t, false);
  check(detector, "near", frontal_r, cv::Vec3d(-40, -40, 220), false);
  check(detector, "far", frontal_r, cv::Vec3d(-40, -40, 1000), false);

  const cv::Vec3d cut_t(-250, -40, 400);
  const std::array<cv::Point2d, 4> cut_truth = project_plate(model, frontal_r, cut_t, camera, dist);
  expect(cut_truth[0].x < 0 && cut_truth[1].x > 30, "cut fixture leaves the notch on screen");
  check(detector, "left edge off screen", frontal_r, cut_t, false);

  // The right corner is the notch. Three outer lines can still lock while
  // that station is outside the image, and the along-edge scale is then free.
  // A square is not published until the notch is actually in the picture.
  MarkerDetector notch_off;
  expect(notch_off.load(model_path, params_path, camera, dist, error), "reload notch off");
  const cv::Vec3d notch_off_t(144, -40, 360);
  const std::array<cv::Point2d, 4> notch_off_truth =
      project_plate(model, frontal_r, notch_off_t, camera, dist);
  expect(notch_off_truth[0].x > 0 && notch_off_truth[0].x < size.width && notch_off_truth[1].x > size.width,
         "notch fixture keeps the left corner and leaves the notch outside");
  const Detection notch_off_detection =
      notch_off.detect(render(model, frontal_r, notch_off_t, camera, dist, size, false));
  expect(!notch_off_detection.found, "notch outside the image is not a pose");

  MarkerDetector solid;
  expect(solid.load(model_path, params_path, camera, dist, error), "reload");
  check(solid, "filled notch is not the marker", frontal_r, frontal_t, true);

  MarkerDetector clutter_detector;
  expect(clutter_detector.load(model_path, params_path, camera, dist, error), "reload clutter");
  cv::Mat clutter = render(model, frontal_r, frontal_t, camera, dist, size, false);
  std::vector<cv::Point> hex = {{30, 640}, {70, 620}, {110, 650}, {90, 710}, {40, 720}, {15, 680}};
  cv::fillConvexPoly(clutter, hex, {255, 255, 255}, cv::LINE_8);
  cv::rectangle(clutter, {820, 40, 70, 40}, {255, 255, 255}, cv::FILLED);
  near_quad(clutter_detector.detect(clutter), project_plate(model, frontal_r, frontal_t, camera, dist),
            "clutter does not move the plate");

  if (g_fails == 0) {
    std::cout << "all passed\n";
    return 0;
  }
  std::cout << g_fails << " failed\n";
  return 1;
}
