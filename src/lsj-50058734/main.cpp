#include<bits/stdc++.h>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
struct MarkerInfo{
    Point2f corners[4];
    vector<vector<Point>> lightContours;
};
struct BrightPart{
    Rect box;
    Point2f center;
    double area;
    int contourIndex;
};
bool detectMarker(const Mat &image,MarkerInfo &marker){
    marker.lightContours.clear();
    if(image.empty()){
        return 0;
    }
    Mat gray, binary;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 220, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<BrightPart> allPart; // 通过基本面积、尺寸检查的亮块
    vector<BrightPart> parts;   // 进一步通过填充率、宽高比检查的亮块
    for (int contourIndex = 0; contourIndex < static_cast<int>(contours.size());contourIndex++){
        const vector<Point> &contour = contours[contourIndex];
        double area = contourArea(contour);
        if(area<20.0)
            continue;
        Rect box = boundingRect(contour);
        if(box.width<5||box.height<5)
            continue;
        BrightPart part{box, Point2f(box.x + box.width / 2.0F, box.y + box.height / 2.0F), area,
                        contourIndex};
        allPart.push_back(part);
        double fillRatio = area / box.area();
        double shapeRatio = static_cast<double>(box.width) / box.height;
        if (fillRatio < 0.25 || fillRatio > 0.75 ||
            shapeRatio < 0.6 || shapeRatio > 1.5)
            continue;
        parts.push_back(part);
    }
    if(parts.size()<3)
        return false;
    sort(parts.begin(), parts.end(),
         [](const BrightPart &first, const BrightPart &second)
         {
             return first.area > second.area;
         });
    if (parts.size() > 8)
        parts.resize(8);
    Rect brightBox;
    vector<Rect> bestLights;
    vector<int> bestContourIndices;
    double bestScore = -1;
    int consize = static_cast<int> (parts.size());
    for (int a = 0; a < consize;a++){
        for (int b = a+1; b < consize;b++){
            for (int c = b + 1; c < consize;c++)
            { // 每次选出三个不同亮块，逐组检查它们能否组成目标的“左上、左下、右下”三个大白灯。
                const BrightPart *three[3] = {&parts[a], &parts[b], &parts[c]};
                sort(three, three + 3,
                     [](const BrightPart *first, const BrightPart *second)
                     {
                         return first->center.y < second->center.y;
                     });//通过中心点坐标来判断位置
                        // 先确定下面两个亮块的左右顺序
                if (three[1]->center.x > three[2]->center.x)
                {
                    swap(three[1], three[2]);
                }
                const BrightPart &topLeft = *three[0];
                const BrightPart &bottomLeft = *three[1];
                const BrightPart &bottomRight = *three[2];
                double smallestArea = min({topLeft.area, bottomLeft.area, bottomRight.area});
                double largestArea = max({topLeft.area, bottomLeft.area, bottomRight.area});
                if(smallestArea/largestArea<0.65){
                    continue;
                }
                double dx = bottomRight.center.x - bottomLeft.center.x;
                double dy = bottomLeft.center.y - topLeft.center.y;
                if(dx<=0||dy<=0||dx/dy<0.7||dx/dy>1.4){
                    continue;
                }
                if(abs(topLeft.center.x-bottomLeft.center.x)>dx*0.25||abs(bottomLeft.center.y-bottomRight.center.y)>dy*0.25)
                    continue;
                if (topLeft.box.width / dx < 0.3 || topLeft.box.width / dx > 0.8)
                    continue;
                vector<Rect> currentLights = {topLeft.box, bottomLeft.box, bottomRight.box};
                vector<int> currentContourIndices = {topLeft.contourIndex, bottomLeft.contourIndex, bottomRight.contourIndex};
                int smallCount = 0;
                for(const BrightPart &part : allPart){//找右上角三个小灯块
                    if (part.center.x > topLeft.center.x + dx * 0.5 &&
                        part.center.x < bottomRight.center.x + dx * 0.6 &&
                        part.center.y > topLeft.center.y - dy * 0.3 &&
                        part.center.y < bottomRight.center.y - dy * 0.35 &&
                        part.area < smallestArea * 0.6)
                    {
                        smallCount++;
                        currentLights.push_back(part.box);
                        currentContourIndices.push_back(part.contourIndex);
                    }
                }
                if(smallCount<2)
                    continue;
                double score = topLeft.area + bottomLeft.area + bottomRight.area;
                if(score>bestScore){
                    bestScore = score;                          // 保存最高分
                    bestLights = currentLights;                 // 保存这组所有白灯的外接矩形
                    bestContourIndices = currentContourIndices; // 保存对应轮廓下标
                    brightBox = bestLights[0];
                    for (const Rect &box : bestLights)
                        brightBox |= box; // 同时包住两个矩形的最小轴对齐矩形
                }
            }
    }
}
    if(bestScore<0){
        return 0;
    }
    for (int contourIndex : bestContourIndices)
        marker.lightContours.push_back(contours[contourIndex]);
    marker.corners[0] = Point2f(brightBox.x, brightBox.y);
    marker.corners[1] = Point2f(brightBox.br().x - 1, brightBox.y);
    marker.corners[2] = Point2f(brightBox.br().x - 1, brightBox.br().y - 1);
    marker.corners[3] = Point2f(brightBox.x, brightBox.br().y - 1);//保存轮廓四点
    return 1;
}
int main()
{
    string inputPath = "data/raw/marker_video.avi";
    string outputPath = "evidence/marker_result.avi";
    if (filesystem::absolute(inputPath).lexically_normal() ==
        filesystem::absolute(outputPath).lexically_normal())
    {
        cerr << "输出视频不能覆盖输入视频" << endl;
        return 1;
    }
    VideoCapture capture(inputPath);
    Mat image;
    if (!capture.read(image) || image.empty())
    {
        cerr << "视频没有可读取帧：" << inputPath << endl;
        return 1;
    }
    double fps = capture.get(CAP_PROP_FPS);
    if(!isfinite(fps)||fps<=0.0){
        fps = 60;
    }
    filesystem::path outputFile(outputPath);
    if (!outputFile.parent_path().empty())
        filesystem::create_directories(outputFile.parent_path());
    VideoWriter writer(outputPath,
                       VideoWriter::fourcc('M', 'J', 'P', 'G'),
                       fps, image.size());
    if (!writer.isOpened())
    {
        cerr << "无法创建输出视频：" << outputPath << endl;
        return 1;
    }
    int frameIndex = 0;
    int detectedFrames = 0;
    do{
        Mat result = image.clone();
        MarkerInfo marker;
        bool found = detectMarker(image, marker);//判断是否能在当前画面中找到发光图形
        if(found){
            for (int corner = 0; corner < 4; corner++)
            {
                line(result, marker.corners[corner],
                     marker.corners[(corner + 1) % 4],
                     Scalar(0, 255, 0), 2, LINE_AA);
                circle(result, marker.corners[corner], 4,
                       Scalar(0, 0, 255), FILLED);
                Point textPoint(cvRound(marker.corners[corner].x + 5),
                                cvRound(marker.corners[corner].y - 5));
                putText(result, to_string(corner), textPoint,
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                
            }
            drawContours(result, marker.lightContours, -1,
                         Scalar(0, 100,255), 2, LINE_AA);
            putText(result, "DETECTED", Point(20, 45),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255,0 ), 2);
        }else{
            
            putText(result, "NOT DETECTED", Point(20, 45),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }
        writer.write(result);

    } while (capture.read(image) && !image.empty());
    return 0;
}