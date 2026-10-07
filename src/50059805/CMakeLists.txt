cmake_minimum_required(VERSION 3.10)

project(RM_AUTO_AIM)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable ( RM_AUTO_AIM 
  src/main.cpp 
  src/MarkerDetector.cpp 
  src/Visualizer.cpp 
  src/VideoProcessor.cpp
  src/Calibrator.cpp
  src/PoseEstimator.cpp
)

target_link_libraries(RM_AUTO_AIM ${OpenCV_LIBS})