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

在仓库根目录执行。不写 `--output-dir` 和 `--output-video` 时，只在终端打印统计。

```bash
build/aethyria-50060127/marker_detect --input data/raw/marker_video.avi
```

| 参数 | 作用 |
| --- | --- |
| `--input` | 要读的视频。帧率取容器元数据 |
| `--output-dir` | 把选中的帧写成 `frame_` 加四位帧号的 PNG。没写 `--save-every` 时每帧都写 |
| `--save-every` | 正整数间隔，从窗口第一帧算起，第一帧一定保存。必须和 `--output-dir` 一起用 |
| `--output-video` | 把选中的每一帧写成标注视频，编码 `avc1`，帧率取容器元数据。可以和 PNG 同时写 |
| `--from` | 闭区间起点，默认 0。更早的帧只为接上跨帧记录而检测，不打印、不保存、不计入汇总 |
| `--to` | 闭区间终点。省略则读到视频结束。读完这一帧就停 |
| `--calibrate` | 用 `--input` 估计内参 |
| `--camera` | 标定结果路径，默认 `src/aethyria-50060127/camera.yaml`。只和 `--calibrate` 一起用，检测不读它 |
| `--help` | 打印用法 |

检测命令写了 `--camera`，或标定命令写了输出、间隔、窗口，都会失败。`--save-every` 小于 1、帧号为负、`--to` 早于 `--from`，同样失败。

每隔 20 帧存一张：

```bash
build/aethyria-50060127/marker_detect \
  --input data/raw/marker_video.avi \
  --output-dir output/aethyria-50060127/frames \
  --save-every 20
```

终端按连续段打印 `detected 起-止 support N` 或 `undetected 起-止`。支持数是这一帧怎么定出板面：4 是四条外缘共同确定的单应；3 是三只完整实心灯质心的仿射；2 是两只灯的相似，或只剩一只灯时沿用上一帧的朝向并平移；1 是单灯。上一帧只用来把灯对上号。两只或三只对上的灯用这一帧的质心重新拟合。四条外缘都量到时，用它们的交点换成一个单应，这条结果优先于没有锁住外缘的灯姿态。贴到图像边界的轮廓不提供质心。汇总行里的 `homography`、`affine`、`similarity`、`single` 依次对应支持数 4、3、2、1。

画出的四边形就是这一帧留下的那个姿态。没有假设通过时不画上一帧的框。

测试视频是 `data/raw/marker_video.avi`：1440×1080，1676 帧，约 23.8 秒。本机从容器读到的帧率是 70.4083。这台机器的 Release 构建可以在半分钟内跑完。实心 L 的模板只光栅化一次，之后每条轮廓用位图重叠来打分。

标定视频 `data/raw/calibration_video.avi` 已用来估计内参，结果在 `camera.yaml`。物体单位是圆心间距。这一版检测不读取内参。位姿估计没有做。

## 附上的连续帧

两组都是原分辨率、逐帧保存，约 48 MB。下面两条命令的输出和目录里的文件一致。

```bash
build/aethyria-50060127/marker_detect \
  --input data/raw/marker_video.avi \
  --output-dir src/aethyria-50060127/frames/right-exit \
  --from 0 --to 60

build/aethyria-50060127/marker_detect \
  --input data/raw/marker_video.avi \
  --output-dir src/aethyria-50060127/frames/left-exit \
  --from 852 --to 876
```

- `frames/right-exit/`：第 0 帧到第 60 帧。开头是整块灯板，接着从右侧出画，灯被切成条之后变为未检出。
- `frames/left-exit/`：第 852 帧到第 876 帧。灯板从左侧离开，右侧灯还在时框延伸到画面外，贴边之后变为未检出。`--from` 之前的帧已经把跨帧记录接上，所以这 25 张和从第 0 帧连续跑到这里相同。

整段标注视频可以写到仓库忽略的 `output/`：

```bash
build/aethyria-50060127/marker_detect \
  --input data/raw/marker_video.avi \
  --output-video output/aethyria-50060127/annotated.mp4
```

全片统计和参数写在 `REPORT.md`。仓库忽略 `*.mp4`，提交里仍然是上面的连续帧。

## 画面上的角点

落在画面内的角画环并标注：`LT` 黄、`RT` 青、`RB` 品红、`LB` 橙。角在画面外时不画字，框的可见部分仍画到图像边界。部分出画时，没进画面的角可以落在图像坐标以外，这是这一帧算出来的板角。
