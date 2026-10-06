# 校内赛标志物识别

使用 C++17 和 OpenCV，通过灰度阈值分割、轮廓筛选、几何组合与局部碎片检查，在连续视频帧中识别标志物，绘制目标区域和四个顶点。

## 环境

| 项目 | 实测版本 |
| --- | --- |
| 系统 | Ubuntu 22.04.5 LTS |
| 编译器 | GCC / g++ 11.4.0 |
| C++ 标准 | C++17 |
| CMake | 3.22.1 |
| OpenCV | 4.10.0 |

以上版本来自开发机器查询。当前实现需要带 GUI 的 Linux 环境；程序使用 `imshow` 显示结果。

## OpenCV 4.10.0 安装

开发环境使用官方 `opencv` 和 `opencv_contrib` 4.10.0 源码，通过 CMake 的 Release 配置编译，安装到 `/usr/local`。下面按该安装路线整理复现命令；本作业仅使用 C++，因此关闭 Python 绑定、示例和测试的构建。这些命令用于新环境配置，已有可用 OpenCV 4.10.0 的机器无需重复安装。

### 1. 安装编译、图形界面和视频依赖

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config wget unzip \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev libv4l-dev \
    libxvidcore-dev libx264-dev libtbb-dev
```

### 2. 下载指定版本源码

在工程目录以外准备依赖，避免将源码和编译产物混入作业提交：

```bash
mkdir -p ~/marker-deps
cd ~/marker-deps
wget -O opencv-4.10.0.zip https://github.com/opencv/opencv/archive/4.10.0.zip
wget -O opencv_contrib-4.10.0.zip https://github.com/opencv/opencv_contrib/archive/4.10.0.zip
unzip opencv-4.10.0.zip
unzip opencv_contrib-4.10.0.zip
```

### 3. 配置、编译与安装

```bash
cmake -S opencv-4.10.0 -B opencv-build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH="$PWD/opencv_contrib-4.10.0/modules" \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DWITH_GTK=ON \
    -DWITH_FFMPEG=ON
cmake --build opencv-build -j2
sudo cmake --install opencv-build
sudo ldconfig
pkg-config --modversion opencv4
```

最后应显示 `4.10.0`。配置输出中应确认 GTK 和 FFMPEG 已启用，供窗口显示和视频读写使用。`opencv_contrib` 延续原开发环境的安装路线，本检测程序不直接使用其扩展模块。

安装流程参考 [OpenCV 4.10.0 官方 Linux 安装文档](https://docs.opencv.org/4.10.0/d7/d9f/tutorial_linux_install.html)。上述整理命令尚未在另一台干净机器上完整执行；当前提交工程已在本机 OpenCV 4.10.0 环境中独立配置并编译成功。

查看环境版本：

```bash
cat /etc/os-release
g++ --version
cmake --version
pkg-config --modversion opencv4
```

## 目录与编译

个人工程位于仓库的 `src/50059740/` 中，包括：

- `CMakeLists.txt`
- `src/test.cpp`
- `README.md`
- `REPORT.md`
- `evidence/result.avi`：不超过 30 秒的结果视频，或替换为连续帧截图。

从仓库根目录进入个人工程目录后执行：

```bash
cd src/50059740
cmake -S . -B build
cmake --build build -j2
```

CMake 通过 `find_package(OpenCV ...)` 查找依赖，不写死本机头文件和库路径。当前配置指定源文件为 `src/test.cpp`，文件名和目录必须一致。

## 运行

程序接受两个参数：输入视频路径或 `camera`，以及输出 AVI 路径。

```bash
./build/marker_detector /absolute/path/to/marker_video.avi result.avi
```

默认摄像头入口：

```bash
./build/marker_detector camera camera_result.avi
```

摄像头分支使用整数设备索引 `cap.open(0)` 打开默认摄像头。由于开发时没有摄像头，该入口未实测。

输入和输出请使用不同的文件路径。输出采用 MJPG 编码，保持输入帧尺寸；输入帧率不可用时使用 30 FPS。按 `Esc` 结束，按 `S` 保存当前原图、二值图及标注图到工作目录。

## 结果说明

- 黄色四边形：通过整体几何和右上碎片检查的目标区域。
- 蓝色点和黄色标签：四个外缘顶点，按 `LT、RT、RB、LB` 输出和连接。
- 绿色小框：通过单区域筛选的候选；红色小框：被筛掉的区域。这些调试框不代表识别到完整目标。
- `Detected`：当前帧至少存在一个通过检查的整体候选。
- `Not detected`：当前帧没有通过检查的整体候选。

坐标以当前输入原图左上角为原点，x 向右、y 向下，单位为像素。主流程没有缩放或裁剪整帧；右上局部窗口中的轮廓点通过加上窗口原点恢复到原图坐标。目标四边形描述的是发光图案外缘，不是黑色背板外缘。

终端输出帧编号、单区域候选数和整体候选数。目前不打印四点数值；四点通过程序内部数组与画面标签表达。

## 验证与提交

本机已完成个人提交目录的独立 CMake 配置与编译。视频人工检查结果：完整视频中目标连续显示、轻微晃动时标签稳定、目标离开后旧框消失；保存的视频正常包含标注。以上为人工观察，尚无准确率或角点误差统计。摄像头未实测，相机标定和位姿估计未实现。

识别结果视频位于 `evidence/result.avi`，时长不超过 30 秒。不要提交 `build/`、可执行文件、目标文件、编辑器配置、完整数据集及大体积中间结果。
