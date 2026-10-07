# RM 校内赛标志物识别 - 视觉算法作业 
## 1. 项目简介
本项目为 RobotMaster 校内赛视觉算法作业。基于 C++ 和 OpenCV 实现了相机标定、位姿解算和标记检测等功能。项目采用面向对象（OOP）架构设计，包含必做部分（标志物检测、时序稳定、状态机管理）与挑战部分（圆点标定板相机标定、PnP 位姿估计）。

## 2. 开发与运行环境
本项目在以下 Linux 环境下开发和测试：
- 操作系统: Ubuntu 20.04
- 编译器: g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 (支持 C++17)
- 构建工具: CMake 3.28.3
- 视觉库: OpenCV 4.6.0

## 3. 项目结构
```
src/50059805/
├── CMakeLists.txt
├── README.md
├── REPORT.md
├── include/
│   ├── MarkerDetector.h   // 标志物检测类（预处理、轮廓提取、顶点排序、边缘判断）
│   ├── Visualizer.h       // 可视化类（绘制外接框、顶点、状态提示）
│   ├── VideoProcessor.h   // 视频处理与状态机类（防闪烁、状态保持）
│   ├── Calibrator.h       // 相机标定类（圆心提取、内参解算、参数保存）
│   └── PoseEstimator.h    // PnP位姿估计类（3D-2D投影、坐标轴绘制）
└── src/
    ├── main.cpp           // 调度入口（支持 mandatory 和 challenge 两种模式）
    ├── MarkerDetector.cpp
    ├── Visualizer.cpp
    ├── VideoProcessor.cpp
    ├── Calibrator.cpp
    └── PoseEstimator.cpp
```
## 4. 编译指南
本项目使用标准的 CMake 流程进行构建。请在个人目录（src/50059805/）下执行以下命令：
``` bash
cmake -S . -B build
cmake --build build
```
编译成功后，会在 build/ 目录下生成可执行文件 RM_AUTO_AIM。

## 5. 运行指南
前置准备：确保测试视频已放在仓库的 data/raw/ 目录下（calibration_video.avi 和 marker_video.avi）
### 5.1 运行必做部分（标志物检测）
``` bash
# 在build目录下运行以下命令：
./RM_AUTO_AIM mandatory
```
输出：屏幕显示检测结果。目标完整时显示绿色框，目标丢失时显示红色 Target Lost。
退出：按 ESC 键退出。
### 5.2 运行挑战部分（相机标定与位姿解算）
``` bash
# 在build目录下运行以下命令：
./RM_AUTO_AIM challenge
```
输出：屏幕显示标志物检测框，并在目标完整可见时绘制红（X）、绿（Y）、蓝（Z）三色坐标轴及距离数据。
退出：按 ESC 键退出。