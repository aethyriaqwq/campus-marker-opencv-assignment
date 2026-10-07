# 校内赛 MARK 标志物识别

作者：杨冉星　学号：50032560

本项目使用 C++17、OpenCV C++ 接口和 CMake，实现连续帧中的传统视觉检测。支持视频文件和摄像头输入；输出检测状态、目标区域、LT/RT/RB/LB 四角、可视化视频和逐帧 CSV。未使用深度学习模型或在线识别服务。

## 1. 环境与安装

实际验证环境：WSL2 Ubuntu 24.04.3 LTS、GCC 13.3.0、CMake 3.28.3。分别验证了 Ubuntu 的 OpenCV 4.6.0 和本机已有的 OpenCV 4.12.0。采用 C++17；只依赖标准库和 OpenCV 的 core、imgproc、imgcodecs、videoio、highgui 模块。

Ubuntu 24.04 安装：

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev
```

不需要 Python、额外模型、CUDA 或本机硬编码路径。通过 `find_package(OpenCV 4 REQUIRED ...)` 自动查找并链接依赖。在无桌面的 Linux 环境中默认不打开窗口。

## 2. 编译与测试

以下命令全部从仓库根目录执行：

```bash
cmake -S src/50032560 -B build/50032560 -DCMAKE_BUILD_TYPE=Release
cmake --build build/50032560 --parallel
ctest --test-dir build/50032560 --output-on-failure
```

`marker_tests` 验证透视目标、不同处理分辨率的原图坐标恢复、四角顺序、小幅运动、目标消失、重新出现、缺角遮挡、纯白背景、亮方块干扰和双目标。当仓库测试视频存在时，还会执行实际视频的回归检查，涵盖普通目标、重复候选抑制和空场景。

## 3. 运行

处理完整测试视频并导出所有帧：

```bash
./build/50032560/marker \
  --input data/raw/marker_video.avi \
  --output outputs/50032560/demo.avi \
  --csv outputs/50032560/frames.csv
```

MJPG/AVI 输出保留输入分辨率和容器帧率。仓库视频为 1440×1080、1676 帧、约 70.408336 FPS，总时长约 23.8 秒，符合不超过 30 秒的演示要求。程序读取元数据，不硬编码这些视频属性。摄像头帧率无效时以 30 FPS 写视频并提示；离线视频的 CSV 时间按帧号/FPS 计算，适用于本作业的固定帧率视频。

实时摄像头（本地 Linux 上摄像头编号通常从 0 开始）：

```bash
./build/50032560/marker --camera 0 --show --csv outputs/50032560/camera.csv
```

窗口中按 Esc 或 q 退出。摄像头设备需由运行环境提供；本次验证使用仓库连续视频，未做现场摄像头测试。摄像头 CSV 记录从启动起的单调时钟时间。

复现提交的连续进出画面截图（帧号从 0 开始）：

```bash
./build/50032560/marker \
  --input data/raw/marker_video.avi \
  --start 30 --max-frames 10 \
  --snapshots outputs/50032560/consecutive \
  --snapshot-start 30 --snapshot-count 10
```

常用参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--input` / `--camera` | 必选其一 | 文件路径 / 摄像头编号 |
| `--width` | 960 | 检测工作图最大宽度；0 为原图，不上采样 |
| `--threshold` | 180 | 灰度分割阈值，范围 1～254 |
| `--min-score` | 0.64 | 透视图案 F1 分数最低值 |
| `--alpha` | 0.75 | 平滑时当前观测的权重；1 表示不平滑 |
| `--start` | 0 | 视频起始帧；摄像头不支持跳帧 |
| `--max-frames` | 0 | 最大处理帧数；0 表示持续读取至结束 |
| `--show` | 关闭 | 打开可视化窗口 |
| `--snapshots` | 无 | 截图目录，与起始帧、截图数量配合 |

完整参数帮助：`./build/50032560/marker --help`。长视频可通过 `--max-frames` 限制演示长度，例如 30 FPS 的视频取 900 帧。

## 4. 输出定义

- 坐标单位为**原始输入图像像素**，原点在左上角，x 向右、y 向下。检测时的缩放会按实际宽高分别还原。
- 目标区域定义为**四角灯带外缘围成的四边形**，对应尺寸图标注的 80×80 发光图案范围，覆盖识别图案；不把外部黑色安装外壳作为关键点边界。所有输出、模板和文档使用同一定义。
- 四角顺序为图像相对的 LT、RT、RB、LB。点绕中心排序，选 x+y 最小点作为 LT，并顺时针排列。若大角度旋转越过对角线，图像相对角标可能更换物理角；本项目不将其用于物体姿态估计。
- 可视化：绿色四边形，LT 绿、RT 黄、RB 蓝、LB 紫。`NOT DETECTED` 表示“未检测到”，表示当前帧没有满足四角完整性与图案校验的目标。
- CSV：`frame,timestamp_ms,detected,id,score,bbox_x,bbox_y,bbox_w,bbox_h,lt_x,lt_y,rt_x,rt_y,rb_x,rb_y,lb_x,lb_y`。
- 无目标帧也有一行，`detected=0,id=-1,score=0`，坐标字段留空；多目标时每个目标一行，帧号相同。`bbox` 是四角的轴对齐外接矩形。
- `score` 是当前帧与设计图案的像素 F1 相似度，不是经过统计校准的置信概率。ID 仅保持相邻帧关联，目标消失后再次出现会获得新 ID。
- **当前帧校验失败立即不输出该目标**，不会因为平滑或上一帧检测到而延用旧框。

## 5. 交付文件

| 文件 | 用途 |
|---|---|
| `detector.hpp`、`detector.cpp` | 检测、四角拟合、去重、当前帧验证与有限平滑 |
| `main.cpp` | 视频/摄像头入口、命令行、CSV、截图和视频输出 |
| `tests.cpp` | 合成场景及仓库视频回归测试 |
| `CMakeLists.txt` | 独立配置、编译、CTest |
| `REPORT.md` | 原理、参数、验证结果、失败案例与改进 |
| `evidence/frame_*.jpg` | 原始分辨率连续结果截图 |
| `evidence/frames.csv` | 完整 1676 帧的实际检测记录 |
| `evidence/validation.txt` | 编译环境、回归与运行摘要 |

按作业要求，提交包以连续截图作为验收材料；完整结果视频可由上述命令生成。本地另提供完整演示视频。构建目录、可执行文件、原始数据副本和大型中间产物均不纳入个人提交目录。

必做部分已实现。选做的相机标定和 PnP 位姿估计未实现，不提供虚构内参或距离结果。
