cmake_minimum_required(VERSION 3.16)
project(campus_marker LANGUAGES CXX)

find_package(OpenCV 4 REQUIRED)

add_executable(campus 
    main.cpp
)
set_target_properties(campus PROPERTIES
    CXX_EXTENSIONS OFF
)
target_include_directories(campus PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(campus PRIVATE ${OpenCV_LIBS})