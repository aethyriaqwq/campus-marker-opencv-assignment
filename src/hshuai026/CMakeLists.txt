cmake_minimum_required(VERSION 3.16)
project(hshuai026_campus_marker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# core/imgproc/imgcodecs 图像处理与读写、videoio 视频读写、
# calib3d 标定与 PnP、features2d 圆点检测
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio calib3d
                                        features2d)

# 公共模块：检测、跟踪、位姿、可视化、视频工具
add_library(marker_core STATIC
    marker_detector.cpp
    marker_tracker.cpp
    pose_estimator.cpp
    overlay.cpp
    video_utils.cpp
)
target_include_directories(marker_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                              ${OpenCV_INCLUDE_DIRS})
target_link_libraries(marker_core PUBLIC ${OpenCV_LIBS})

# 主程序：检测 + 跟踪 +（可选）位姿，输出统一的一套结果
add_executable(campus_marker main.cpp)
target_link_libraries(campus_marker PRIVATE marker_core)

# 相机标定（圆点阵列标定板）
add_executable(calib_dots calib_dots.cpp)
target_include_directories(calib_dots PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(calib_dots PRIVATE ${OpenCV_LIBS})

# 时序滤波方案评测（纯标准库）
add_executable(filter_eval filter_eval.cpp)
