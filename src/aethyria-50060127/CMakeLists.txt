cmake_minimum_required(VERSION 3.16)
project(aethyria_marker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio calib3d)

add_executable(marker_detect
  src/main.cpp
  src/marker_model.cpp
  src/detector.cpp
  src/calibrate.cpp)
target_include_directories(marker_detect PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(marker_detect PRIVATE ${OpenCV_LIBS})
target_compile_options(marker_detect PRIVATE -Wall -Wextra -Wpedantic)

add_executable(marker_geometry_test
  src/geometry_test.cpp
  src/marker_model.cpp
  src/detector.cpp)
target_include_directories(marker_geometry_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(marker_geometry_test PRIVATE ${OpenCV_LIBS})
target_compile_options(marker_geometry_test PRIVATE -Wall -Wextra -Wpedantic)
