#include <bits/stdc++.h>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main(){
    string inputPath = "data/raw/calibration_video.avi";
    double circleSpacing = 0.03;
    string outputFilename = "outputs/calibration_outputs.yml";
    filesystem::path outputPath(outputFilename);
    VideoCapture capture(inputPath);
    if (!capture.isOpened())
    {
        cerr << "无法打开标定视频：" << inputPath << endl;
        return 1;
    }
    const int frameCount = cvRound(capture.get(CAP_PROP_FRAME_COUNT));
    const Size patternSize(7, 7); // 7×7 个圆心，不是 7×7 个方格
    vector<Point3f> boardPoints;
    for (int row = 0; row < patternSize.height;row++)
        for (int col = 0; col < patternSize.width;col++)
            boardPoints.push_back(Point3f(col * circleSpacing, row * circleSpacing, 0.0F));//生成标定板上圆形的三维坐标
    vector<vector<Point3f>> objectPoints;// 所有有效抽样帧的物点
    vector<vector<Point2f>> imagePoints;// 所有有效抽样帧的像点
    vector<int> frameIndices;
    vector<Point2f> boardCenters;
    Size imageSize;                          // 标定图像的宽度、高度，单位为像素
    const int step = max(1, frameCount / 14);//约取十四帧，不取一帧是防止镜头畸变
    for (int frameIndex = frameCount / 35; frameIndex < frameCount;frameIndex+=step){
        capture.set(CAP_PROP_POS_FRAMES, frameIndex); // set() 请求将下一次读取位置定位到 frameIndex 帧
        Mat image;
        if(!capture.read(image)||image.empty())
            continue;
        if(imageSize.empty())
            imageSize = image.size();
        if(image.size()!=imageSize){
            //保持分辨率一致
            cerr << "画面尺寸发生变化" << endl;
            return 1;
        }
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        vector<Point2f> centers;
        if(!findCirclesGrid(gray,patternSize,centers,CALIB_CB_SYMMETRIC_GRID))
        { // findCirclesGrid() 查找完整圆点阵，将有序圆心写入 centers，返回是否找到
            continue;
        }
        cornerSubPix(gray, centers, Size(5, 5), Size(-1, -1), // cornerSubPix() 对检测点做亚像素细化；搜索窗口半径为 5 像素，(-1,-1) 表示不排除中心区域
                     TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
                                  30, 0.01)); // TermCriteria() 设置停止条件：最多 30 次迭代，或点位置变化不超过 0.01 像素
        Point2f meanCenter(0.0F, 0.0F);
        for(const Point2f &point:centers){
            meanCenter += point;
        }
        meanCenter *= 1.0F / centers.size();
        boardCenters.push_back(meanCenter);
        objectPoints.push_back(boardPoints);
        imagePoints.push_back(centers);
        frameIndices.push_back(frameIndex);
    }
    Rect2f centerRange = boundingRect(boardCenters);
    Mat cameraMatrix, distortion; // cameraMatrix 是 3×3 内参矩阵（fx、fy、cx、cy 等）；distortion 是镜头畸变系数（默认 k1、k2、p1、p2、k3）
    vector<Mat> rotations, translations; // 每帧的外参：rotations 为旋转向量（轴角，弧度），translations 为平移向量（米），均将板坐标变换到相机坐标
    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, // calibrateCamera() 根据多帧物点与像点求相机参数；rms 是整体重投影均方根误差，单位为像素
                                 cameraMatrix, distortion,             // 两个输出参数：内参矩阵、畸变系数
                                 rotations, translations);             // 两个输出参数：每帧标定板的旋转、平移；本程序计算但不保存这些外参
    if (!isfinite(rms) || !checkRange(cameraMatrix) ||                 // isfinite() 检查误差为有限数；checkRange() 检查矩阵元素没有 NaN 或无穷等异常值
        !checkRange(distortion) || cameraMatrix.at<double>(0, 0) <= 0 || // at<double>(0,0) 读取内参矩阵的 fx，要求横向焦距参数为正
        cameraMatrix.at<double>(1, 1) <= 0)                              // at<double>(1,1) 读取 fy，要求纵向焦距参数为正
    {
        cerr << "标定计算没有得到有效的相机参数" << endl;
        return 1;                                         
    }
    if (!outputPath.parent_path().empty())
        filesystem::create_directories(outputPath.parent_path());
    FileStorage file(outputPath.string(), FileStorage::WRITE);
    if (!file.isOpened())
    {
        cerr << "无法保存标定文件：" << outputPath << endl;
        return 1;
    }
    file << "camera_matrix" << cameraMatrix;         // 以 camera_matrix 为字段名保存内参；FileStorage 的 << 写入字段名和对应数据
    file << "distortion_coefficients" << distortion; // 保存镜头畸变系数，后续去畸变或 PnP 可读取
    file << "image_width" << imageSize.width;        // 保存本次标定图像的宽度，单位为像素
    file << "image_height" << imageSize.height;      // 保存本次标定图像的高度，单位为像素
    file << "board_columns" << patternSize.width;    // 保存圆点阵列数，本次为 7
    file << "board_rows" << patternSize.height;      // 保存圆点阵行数，本次为 7
    file << "circle_spacing" << circleSpacing;       // 保存生成物点时使用的相邻圆心间距，本次为 0.03
    file << "length_unit" << "m";                    // 保存长度单位 m，即米；说明 circle_spacing 的单位
    file << "rms_pixels" << rms;                     // 保存整体重投影误差；不是实际测距误差，较小也不代表所有场景精度可靠
    file << "sample_frames" << frameIndices;         // 保存参与本次标定的原视频帧下标，便于回查素材
    file.release();
    cout << "成功标定帧：" << frameIndices.size()
         << "，重投影 RMS：" << fixed << setprecision(3) << rms
         << " 像素，输出：" << outputPath << endl;
    cout << "圆心间距：" << circleSpacing << " 米" << endl; 
    return 0;                                               
    return 0;
}
