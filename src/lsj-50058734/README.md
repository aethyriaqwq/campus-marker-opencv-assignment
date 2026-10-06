# 校内赛标志物识别与相机标定

个人标识：`lsj_50058734`。提交目录按仓库命名约定使用 `lsj-50058734`。

本项目使用 C++17 和 OpenCV C++ 接口，通过 CMake 在 Linux 下编译。白灯识别与相机标定分别运行；两个程序都使用无参数的 `int main()`。方法与学习记录见 [REPORT.md](REPORT.md)。

## 目录

```text
src/lsj-50058734/
├── main.cpp                         # 白灯检测、目标区域与四点绘制、连续视频输出
├── calibrate.cpp                    # 7×7 对称圆点阵相机标定
├── CMakeLists.txt                   # 独立构建 marker 和 calibrate
├── README.md                        # 环境、构建、运行和提交说明
├── REPORT.md                        # 保留原实验报告
└── evidence/
    ├── marker_result.avi            # 整理前已有的识别结果视频
    └── calibration_outputs.yml      # 整理前已有的标定结果
```
## 环境与依赖

| 项目 | 版本 |
| --- | --- |
| Linux | Ubuntu 24.04.4 LTS，WSL Ubuntu-24.04 |
| 编译器 | GCC / g++ 13.3.0 |
| C++ 标准 | C++17 |
| CMake | 3.28.3 |
| OpenCV | 4.6.0 |

Ubuntu 下安装依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev pkg-config
```

版本核对命令：

```bash
lsb_release -ds
g++ --version
cmake --version
pkg-config --modversion opencv4
```

## CMake 配置与编译

同时编译两个程序：

```bash
cmake -S src/lsj-50058734 -B build/lsj-50058734 -DCMAKE_BUILD_TYPE=Release
cmake --build build/lsj-50058734 --parallel
```

只编译识别程序或只编译标定程序，先执行上述配置命令，再任选一条：

```bash
cmake --build build/lsj-50058734 --target marker --parallel
cmake --build build/lsj-50058734 --target calibrate --parallel
```

## 运行白灯识别


```bash
./build/lsj-50058734/marker
```
## 运行标定试验


```bash
./build/lsj-50058734/calibrate
```