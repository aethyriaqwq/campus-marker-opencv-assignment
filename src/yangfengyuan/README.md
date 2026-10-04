# 国庆作业1

这个代码用于检测并标注视频中目标是否出现

## 构建与实行

需要 CMake 3.16 及以上版本、C++17 编译器和支持 MP4 读写的 OpenCV 4 开发包。

```bash
cmake -S . -B build
cmake --build build
./build/campus ../../data/raw/marker_video.avi results
```

每段视频会输出以下内容，全部与输入视频保持相同帧率，由 OpenCV 的 `VideoWriter` 直接写为 `mp4v` 编码的 MP4。程序不再进行外部转码；新生成的视频可能无法在 VS Code 内置播放器中直接打开，可使用支持 MPEG-4 Part 2 的播放器查看。

-`results/<视频名>_annotated.mp4`: 原尺寸的最终标注视频。

## 流程与参数
1. **去噪** 中值滤波
2. **二值化** 二值化一张灰度图的经验参数: `200`作为阈值
3. **形态学处理** 目的是将不连续的方框连接，做左右连接和上下连接两个操作。之后再取出噪点. kernel大小: `kernel_vertical(5*60)`用来连接上下断点,`kernel_horizontal(60*5)`用来连接左右断点; `process1`,`process2`是记录处理的中间态; `progress`记录结果
4. **画矩形** 矩形方框筛选条件: 面积小于`30000`即筛掉->一个正常方框的面积约为`60000`
鉴于该视频的特性，每次最多只会有1个矩形框
5. **画角点** 取旋转矩形的顶点`vertices[4]`画点
6. **返回结果** `rotated_rect`为最终处理并返回的结果。