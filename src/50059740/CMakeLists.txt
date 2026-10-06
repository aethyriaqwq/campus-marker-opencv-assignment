cmake_minimum_required(VERSION 3.16)
project(marker_assignment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED
    COMPONENTS core imgproc imgcodecs videoio highgui
)

# ========= marker_detector 目标 =========
add_executable(marker_detector src/test.cpp)
target_include_directories(marker_detector PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(marker_detector PRIVATE
    ${OpenCV_LIBS}
)