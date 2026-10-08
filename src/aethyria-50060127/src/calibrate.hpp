#pragma once

#include <string>

// Estimate K and distortion from the circle-grid video and write them to yaml_path.
// The object unit is one circle-center spacing. Pixel intrinsics do not depend on that scale.
bool calibrate_camera(const std::string& video_path, const std::string& yaml_path);
