# hshuai026：校内赛标志物识别（传统视觉）

> 作者：帅蘅轩（学号 hshuai026）

用 C++17 + OpenCV 4 的传统视觉流程识别 `data/raw/marker_video.avi` 中的
校内赛标志物（暗背景上的白色方形灯组），在原始图像坐标系中输出目标外框和
LT / RT / RB / LB 四个顶点，目标离开画面后给出“未检测到”。

算法原理、参数依据与实测数据见 [REPORT.md](REPORT.md)。

## 0. 目录结构与模块划分

代码按"单一职责"拆成小模块，主程序只负责流程编排，方便单独调试与复现：

| 文件 | 职责 | 被谁使用 |
| --- | --- | --- |
| `main.cpp` | **主程序**：读帧 → 检测 → 跟踪 →（可选）位姿 → 绘制 → 写出；检测与位姿在同一套结果里 | `campus_marker` |
| `marker_detector.hpp/.cpp` | **单帧检测**：亮度阈值 → 形态学 → 几何筛选 → 凸包 → 四点排序 | 必做、位姿 |
| `marker_tracker.hpp/.cpp` | **跨帧跟踪**：滤波、短暂丢失用运动模型预测、持续丢失返回"未检测到" | 必做 |
| `tracking_filter.hpp` | **匀速卡尔曼滤波**（CV 模型）实现，含公式注释与踩坑记录 | 必做、位姿 |
| `overlay.hpp/.cpp` | **可视化**：外框、四个顶点、实时像素坐标、坐标轴字母 | 必做、位姿 |
| `video_utils.hpp/.cpp` | **路径与视频 I/O**：跨目录找素材、摄像头输入、编码器回退、避免旧数据残留 | 必做、位姿 |
| `pose_estimator.hpp/.cpp` | **位姿估计**：读标定参数，用 PnP 求标志物在相机坐标系下的位姿 | 主程序 |
| `calib_dots.cpp` | 圆点阵列标定板相机标定（生成位姿所需的标定参数） | `calib_dots` |
| `filter_eval.cpp` | 滤波方案离线评测（滞后/抖动/保真度），纯标准库 | `filter_eval` |
| `evidence/` | 结果材料：标注视频、标定 YAML、位姿 CSV、截图 | — |

CMake 把上面五个模块编成静态库 `marker_core`，主程序与评测工具共用。

**必做与选做不分开**：检测、跟踪、位姿在同一个程序里依次完成，
输出**同一个视频和同一份日志**（视频上同时有外框、四顶点、像素坐标、
坐标轴和毫米级位姿数值；日志里同时有像素与毫米两套数据），
不再分成两批结果。

## 1. 依赖环境（本机实测版本）

| 项目 | 版本 |
| --- | --- |
| 系统 | Ubuntu 24.04.5 LTS |
| 编译器 | g++ 13.3.0（C++17） |
| 构建工具 | CMake 3.28.3 |
| 图像库 | OpenCV 4.6.0 |

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopencv-dev
```

## 2. 编译（从仓库根目录执行）

```bash
cmake -S src/hshuai026 -B build/hshuai026 -DCMAKE_BUILD_TYPE=Release
cmake --build build/hshuai026 --parallel
```

`CMakeLists.txt` 通过 `find_package(OpenCV 4 REQUIRED ...)` 查找 OpenCV，
不写死任何本机路径；只修改 `src/hshuai026/` 内的文件。

## 3. 运行

```bash
# 只做检测（必做部分）
./build/hshuai026/campus_marker data/raw/marker_video.avi src/hshuai026/evidence

# 检测 + 位姿（必做 + 选做，同一套结果）
./build/hshuai026/campus_marker data/raw/marker_video.avi src/hshuai026/evidence \
    src/hshuai026/evidence/camera_calibration.yaml 80
```

参数依次是：输入视频（或摄像头序号如 `0`）、输出目录、标定 yaml、标志物边长 mm。
省略标定 yaml 时只输出检测结果；输出目录省略时默认 `src/hshuai026/results`。

## 4. 输出文件

| 文件 | 说明 |
| --- | --- |
| `marker_video_result.mp4` | 结果视频：目标框 + 四顶点 + 像素坐标 +（有标定时）坐标轴与毫米级位姿 |
| `marker_video_result.csv` | 逐帧日志：像素数据 + 位姿数据（`cx,cy,width,height,angle,x_mm,y_mm,z_mm,distance_mm,rx,ry,rz,reproj_px`） |
| `overview_xxxxx.jpg` | 每 150 帧一张的全程概览截图 |
| `sequence_00250.jpg` ~ `sequence_00279.jpg` | 连续 30 帧截图，用于展示检测连续性 |

注意：仓库根目录 `.gitignore` 会忽略 `results/` 和 `*.mp4`，所以提交进仓库的
验收材料是 JPEG 截图与 CSV，标注视频在本机运行上面的命令即可重新生成。
标注视频按 `avc1 → H264 → mp4v` 的顺序尝试编码：本机 OpenCV 缺少 H.264 编码器时
会自动回退成 MPEG-4 (mp4v)，两种文件都能被 OpenCV 和常见播放器打开。

## 5. 算法流程（对应 `main.cpp` 中的步骤注释）

1. **亮度阈值**：灰度图 `> 200` 取高亮区域。标志物是暗背景上的白色灯组，
   背景均值不到 60、灯面约 250，阈值能稳定分开，且不受色温影响。
2. **形态学处理**：3×3 开运算去掉噪点；51×51 膨胀把分散的灯块合并成一个
   整体候选区域（标志物内部最宽的暗缝约 45 px）。
3. **轮廓 + 几何筛选**：面积 1500~300000 px²、边长 80~900 px、长宽比 ≤ 2.0、
   紧凑度 ≥ 0.30，取面积最大的候选。
4. **外框拟合**：只用候选区域内的原始高亮点做凸包，再拟合最小外接旋转矩形，
   外框因此贴着灯面本身，而不是被膨胀核放大后的轮廓。
5. **四点排序**：先按 x 坐标分左右，再按 y 坐标分上下，输出 LT、RT、RB、LB，
   坐标始终在原始 1440×1080 图像坐标系中。
6. **跨帧稳定**：每个顶点各跑一路**匀速卡尔曼滤波（CV）**（`tracking_filter.hpp`），
   同时估计位置与速度并输出预测位置——既抑制抖动，又不会像 EMA 那样在快速运动时
   产生 v·τ 的稳态滞后；检测失败时用运动模型前推最多 5 帧，超过后立即清空并显示
   “NO TARGET”，不会无限复制上一帧结果（详见 REPORT.md 第 9 节实测对比）。

## 6. 可调参数

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `bright_threshold` | 200 | 高亮阈值，背景更亮时需要调高 |
| `merge_kernel` | 51 | 合并灯块的膨胀核，目标变大/灯块间缝更宽时调大 |
| `min_area` / `max_area` | 1500 / 300000 | 候选面积范围 |
| `min_side` / `max_side` | 80 / 900 | 候选边长范围 |
| `max_aspect` | 2.0 | 外接旋转矩形的长宽比上限 |
| `min_compactness` | 0.30 | 候选面积 / 外接矩形面积，用于排除细长干扰 |
| `max_missed_frames` | 5 | 连续丢失多少帧后判定为“未检测到” |
| `filter_process_noise` | 0.5 | 卡尔曼过程噪声 q，越大越跟随检测 |
| `filter_measurement_noise` | 9.0 | 观测噪声 r（像素²），越大越平滑 |
| `filter_predict_steps` | 0 | 流水线延迟补偿（提前帧数），确有固定延迟时才设 >0 |

## 7. 已知限制

- 阈值根据本作业素材（暗背景 + 白灯）整定，明亮或白色背景的场景需要重新标定；
- 目标部分移出画面时只框住可见部分，框宽会变窄；
- 当前只输出面积最大的一个目标，多标志物同框时只保留一个；
- 未做镜头畸变校正，画面边缘的顶点会有少量畸变偏差；

## 8. 挑战部分：相机标定 + 位姿估计

同一个 CMake 工程还会编译两个挑战部分的可执行文件：

| 可执行文件 | 作用 |
| --- | --- |
| `calib_dots` | 用 `calibration_video.avi`（黑框 + 7×7 圆点阵列标定板）标定内参与畸变 |
| `pose_pnp` | 用标定结果对 `marker_video.avi` 做 PnP 位姿估计（距离/姿态，单位 mm） |

### 8.1 相机标定

```bash
./build/hshuai026/calib_dots data/raw/calibration_video.avi \
    src/hshuai026/evidence/camera_calibration.yaml 30 15 12
```

参数依次是：标定视频、输出 YAML、点间距（mm，仅用于记录尺度）、采样间隔、
最少视角数。程序会：逐帧用 `cv::findCirclesGrid` 检测 7×7 点阵 → 按清晰度与
点阵跨度筛视角 → `cv::calibrateCamera`（固定 k3 的 4 参数畸变模型）→ 按
2×RMS 剔除异常视角后重标定 → 输出内参、畸变系数、重投影误差与点阵检出示意图。

实测：6966 帧中检出 450 次，筛出 45 个视角、剔除 5 个异常后使用 40 个视角，
**重投影误差 RMS = 1.28 px**（平均 0.97 px，最差 3.70 px）。

### 8.2 位姿估计

```bash
# 与必做部分同一条命令：多给一个标定 yaml 就同时输出位姿
./build/hshuai026/campus_marker data/raw/marker_video.avi \
    src/hshuai026/evidence src/hshuai026/evidence/camera_calibration.yaml 80
```

最后一个参数是标志物外轮廓边长（mm），来自 `docs/assets/marker_dimensions.jpg`
主视图的 80×80 标注。程序对每帧：

1. 复用必做部分的检测器得到标志物四角（LT/RT/RB/LB）；
2. 用 `cv::solvePnP`（IPPE，平面靶标）求 `rvec/tvec`；
3. 用 `cv::drawFrameAxes` 画出三轴（**红 = 标志物 X 轴、绿 = Y 轴、蓝 = Z 轴**，
   轴长 80 mm），并在画面上写出 X/Y/Z 距离（**单位 mm**）与该帧重投影误差；
4. 逐帧写入 `pose_log.csv`。

实测：1256/1676 帧检测成功，其中 1281 帧给出位姿结果（含短暂遮挡时的预测帧），
**距离 1017.6 ± 18.0 mm**（983.7 ~ 1062.7 mm），位姿重投影误差平均 **0.67 px**。

### 8.3 挑战部分输出文件

| 文件 | 说明 |
| --- | --- |
| `camera_calibration.yaml` | 内参矩阵、畸变系数、重投影误差、使用的视角数与帧号 |
| `calib_dots_0~3.jpg` | 圆点阵列检出与角点示意图 |
| `marker_video_result.mp4` | 统一结果视频：外框 + 四顶点 + 像素坐标 + 坐标轴（红X/绿Y/蓝Z）+ 毫米级位姿数值（*.mp4 被 .gitignore 忽略） |
| `marker_video_result.csv` | 统一逐帧日志：像素数据与位姿数据在同一行 |
| `overview_*.jpg` / `sequence_*.jpg` | 统一结果截图（含框、顶点、坐标、坐标轴） |
