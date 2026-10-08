# 校内赛标志物识别

个人目录是 `src/aethyria-50060127`。程序对每一帧判断画面里有没有校内赛标志物。有，就在原始图像坐标里给出灯板四角，顺序是图纸上的 `LT`、`RT`、`RB`、`LB`。角点是灯板外缘直线的交点；某一条边在这一帧没有量到时，该角留在灯质心模型上。没有任何假设通过时，画面左上角写 `undetected`，并清掉上一帧的板。Hershey 字体没有中文，所以画面上用这八个字母。

尺寸来自 `docs/assets/marker_dimensions.jpg` 上的整数标注。程序运行时不读这张图。长度都写成图纸单位，或写成这一帧图像上灯距的倍数。

## 环境

Ubuntu 24.04.5，GCC 13.3.0，C++17，CMake 3.28.3，OpenCV 4.6.0。

```bash
sudo apt install build-essential cmake libopencv-dev
```

OpenCV 由 CMake 的 `find_package` 查找，不写安装路径。

## 编译

在仓库根目录执行：

```bash
cmake -S src/aethyria-50060127 -B build/aethyria-50060127 -DCMAKE_BUILD_TYPE=Release
cmake --build build/aethyria-50060127 --parallel
```

可执行文件是 `build/aethyria-50060127/marker_detect`。`build/` 是构建产物，不提交。

## 运行

```bash
build/aethyria-50060127/marker_detect \
  --input data/raw/marker_video.avi \
  --output-dir output/aethyria-50060127/frames \
  --save-every 20
```

| 参数 | 作用 |
| --- | --- |
| `--input` | 视频路径。帧率用容器元数据，程序里不写死 |
| `--from`、`--to` | 闭区间帧号，只统计和保存这一段。`--from` 之前的帧仍会检测，用来接上跨帧记录，但不打印、不保存。`--to` 之后的帧只解码 |
| `--output-dir` | 保存画了结果的 PNG。文件名是 `frame_` 加四位帧号 |
| `--save-every` | 每隔多少帧保存一张。`0` 表示只在终端打印统计 |
| `--calibrate` | 用标定视频估计内参，写到 `--camera`。默认 `src/aethyria-50060127/camera.yaml` |

终端按连续段打印 `detected 起-止 support N` 或 `undetected 起-止`。支持数是这一帧怎么定出板面：3 是三只实心灯质心的仿射，或把上一块板平移到仍对得上的至少三只灯；2 是两只灯的相似，或上一块板只对上一到两只灯；1 是单灯。汇总行里的 `homography`、`affine`、`similarity`、`single` 依次对应支持数 4、3、2、1。这一版不用支持数 4，所以 `homography` 是 0。

画出的四边形跟着这一帧的外缘交点走。整板平移不限制；去掉平移之后，形状每帧大约只变 1 像素。这个限制只作用在画面上，下一帧的灯质心模型仍用没限制过的板。没有假设通过时不画上一帧的框。

测试视频是 `data/raw/marker_video.avi`：1440×1080，1676 帧，约 23.8 秒。本机从容器读到的帧率是 70.4083。这台机器的 Release 构建可以在半分钟内跑完。实心 L 的模板只光栅化一次，之后每条轮廓用位图重叠来打分。

标定视频 `data/raw/calibration_video.avi` 已用来估计内参，结果在 `camera.yaml`。物体单位是圆心间距。这一版检测不读取内参。位姿估计没有做。

## 附上的连续帧

两组都是原分辨率、逐帧保存，约 48 MB。下面两条命令的输出和目录里的文件一致。

```bash
build/aethyria-50060127/marker_detect \
  --input data/raw/marker_video.avi \
  --output-dir src/aethyria-50060127/frames/right-exit \
  --from 0 --to 60 --save-every 1

build/aethyria-50060127/marker_detect \
  --input data/raw/marker_video.avi \
  --output-dir src/aethyria-50060127/frames/left-exit \
  --from 852 --to 876 --save-every 1
```

- `frames/right-exit/`：第 0 帧到第 60 帧。开头是整块灯板，接着从右侧出画，灯被切成条之后变为未检出。
- `frames/left-exit/`：第 852 帧到第 876 帧。灯板从左侧离开，右侧灯还在时框延伸到画面外，贴边之后变为未检出。`--from` 之前的帧已经把跨帧记录接上，所以这 25 张和从第 0 帧连续跑到这里相同。

全片统计和参数写在 `REPORT.md`。仓库忽略 `*.mp4`，这里不附视频。

## 画面上的角点

落在画面内的角画环并标注：`LT` 黄、`RT` 青、`RB` 品红、`LB` 橙。角在画面外时不画字，框的可见部分仍画到图像边界。部分出画时，没进画面的角可以落在图像坐标以外，这是这一帧算出来的板角。
