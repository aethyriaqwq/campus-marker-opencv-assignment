#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// 检查三个亮角中心的几何关系，预测右上位置
bool checkTriple(Point2f lt, Point2f lb, Point2f rb,
                 Point2f& predictedRT)
{
    Point2f u = lt - lb;
    Point2f v = rb - lb;

    float lengthU = static_cast<float>(norm(u));
    float lengthV = static_cast<float>(norm(v));

    if (lengthU < 10.0f || lengthV < 10.0f)
        return false;

    float ratio = max(lengthU, lengthV)
                / min(lengthU, lengthV);

    if (ratio > 2.0f)
        return false;

    float cosine = u.dot(v) / (lengthU * lengthV);

    if (std::abs(cosine) > 0.3f)
        return false;

    predictedRT = lb + u + v;
    return true;
}

// 从一个亮角的旋转矩形中选择外侧顶点
Point2f selectOuterCorner(
    const RotatedRect& rect,
    Point2f markerCenter)
{
    Point2f vertices[4];
    rect.points(vertices);

    // 从整体中心指向该亮角
    Point2f outward = rect.center - markerCenter;

    Point2f selected = vertices[0];
    float bestScore =
        (vertices[0] - markerCenter).dot(outward);

    for (int i = 1; i < 4; ++i) {
        float score =
            (vertices[i] - markerCenter).dot(outward);

        if (score > bestScore) {
            bestScore = score;
            selected = vertices[i];
        }
    }

    return selected;
}

// 将亮角中心组合转换为外缘角点
array<Point2f, 4> locateOuterCorners(
    const RotatedRect& ltRect,
    const RotatedRect& lbRect,
    const RotatedRect& rbRect,
    Point2f predictedRTCenter)
{
    Point2f markerCenter =
        (ltRect.center + lbRect.center +
        rbRect.center + predictedRTCenter)/4.0f;

    Point2f lt = selectOuterCorner(ltRect, markerCenter);
    Point2f lb = selectOuterCorner(lbRect, markerCenter);
    Point2f rb = selectOuterCorner(rbRect, markerCenter);

    // 初版仍采用平行四边形近似
    Point2f rt = lt + rb - lb;

    return {lt, rt, rb, lb};
}

bool measureRightTop(
    const Mat& binary,
    const Rect& search,
    Point2f markerCenter,
    Point2f& measuredRT)
{
    Rect bounds(0, 0, binary.cols, binary.rows);

    if (search.empty() || (search & bounds) != search)
        return false;

    Mat patch = binary(search);

    vector<vector<Point>> fragments;
    findContours(patch, fragments,
                 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Point2f> allPoints;
    int fragmentCount = 0;

    // 按搜索窗口面积设置小碎片过滤阈值
    double minimumArea = max(4.0, search.area() * 0.005);

    for (const auto& fragment : fragments) {
        if (contourArea(fragment) < minimumArea)
            continue;

        ++fragmentCount;

        for (const auto& p : fragment) {
            // 将窗口内坐标转换为原图坐标
            allPoints.push_back(Point2f(
                static_cast<float>(p.x + search.x),
                static_cast<float>(p.y + search.y)
            ));
        }
    }

    // 至少两块有效碎片才继续
    if (fragmentCount < 2)
        return false;

    if (allPoints.size() < 3)
        return false;

    RotatedRect fragmentRect = minAreaRect(allPoints);

    measuredRT = selectOuterCorner(fragmentRect, markerCenter);
    return true;
}

bool validCornerOrder(const array<Point2f, 4>& p)
{
    // 上边：LT 在 RT 左侧；下边：LB 在 RB 左侧
    if (p[0].x >= p[1].x || p[3].x >= p[2].x)
        return false;

    // 左右两侧都应是上点在下点上方
    if (p[0].y >= p[3].y || p[1].y >= p[2].y)
        return false;

    for (int i = 0; i < 4; ++i) {
        Point2f a = p[(i + 1) % 4] - p[i];
        Point2f b = p[(i + 2) % 4] - p[(i + 1) % 4];

        if (norm(a) < 10.0)
            return false;

        // 二维叉积：判断相邻边的转向
        float cross = a.x * b.y - a.y * b.x;

        // 图像坐标中，LT→RT→RB→LB 的凸四边形
        // 每次转向的叉积应为正
        if (cross <= 0.0f)
            return false;
    }

    return true;
}

// 枚举亮角组合，并检查预测位置附近的亮区
vector<array<Point2f, 4>> findMarkerHypotheses(
    const vector<RotatedRect>& candidates,
    const Mat& binary)
{
    vector<array<Point2f, 4>> results;
    int n = static_cast<int>(candidates.size());

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                if (i == j || i == k || j == k)
                    continue;

                Point2f lt = candidates[i].center;
                Point2f lb = candidates[j].center;
                Point2f rb = candidates[k].center;

                // 初版限定目标大致正立
                if (lt.y >= lb.y || rb.x <= lb.x)
                    continue;

                Point2f rt;
                if (!checkTriple(lt, lb, rb, rt))
                    continue;

                float meanSize = (
                    max(candidates[i].size.width,
                        candidates[i].size.height) +
                    max(candidates[j].size.width,
                        candidates[j].size.height) +
                    max(candidates[k].size.width,
                        candidates[k].size.height)
                ) / 3.0f;

                int radius = max(10, cvRound(meanSize * 0.65f));

                Rect search(
                    cvRound(rt.x) - radius,
                    cvRound(rt.y) - radius,
                    2 * radius + 1,
                    2 * radius + 1
                );

                Rect imageBounds(0, 0, binary.cols, binary.rows);

                if ((search & imageBounds) != search)
                    continue;

                Mat patch = binary(search);

                double whitePixels = countNonZero(patch);
                double whiteRatio = whitePixels / search.area();

                if (whiteRatio < 0.04 || whiteRatio > 0.65)
                    continue;

                // 约定输出顺序：LT、RT、RB、LB
                auto corners = locateOuterCorners(
                    candidates[i], candidates[j], candidates[k], rt
                );

                Point2f markerCenter = (lt + rt + rb + lb) / 4.0f;
                Point2f measuredRT;

                if (!measureRightTop(binary, search, markerCenter, measuredRT))
                    continue;

                // 实测点不能偏离预测外角太远
                if (norm(measuredRT - corners[1]) > meanSize * 0.6f)
                    continue;

                corners[1] = measuredRT;

                if (!validCornerOrder(corners))
                    continue;

                results.push_back(corners);
            }
        }
    }

    return results;
}

int main(int argc, char* argv[])
{

    if (argc <= 2) {
        cerr << "Usage: " << argv[0]
             << " <video_path|camera> <output.avi>\n";
        return 1;
    }

    string input = argv[1];
    string outputPath = argv[2];

    VideoCapture cap;

    if (input == "camera")
        cap.open(0);
    else
        cap.open(input);

    if (!cap.isOpened()) {
        cerr << "Cannot open input: " << input << '\n';
        return 1;
    }

    Mat frame;
    int frameIndex = 0;
    

    const char* names[] = {"LT", "RT", "RB", "LB"};

    double fps = cap.get(CAP_PROP_FPS);
    if (fps <= 0.0)
        fps = 30.0;

    Size frameSize(
        cvRound(cap.get(CAP_PROP_FRAME_WIDTH)),
        cvRound(cap.get(CAP_PROP_FRAME_HEIGHT))
    );

    VideoWriter writer(
        outputPath,
        VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps,
        frameSize,
        true
    );

    if (!writer.isOpened()) {
        cerr << "Cannot create output video\n";
        return 1;
    }

    while (cap.read(frame)) {
        if (frame.empty())
            break;

        Mat gray, binary;
        vector<RotatedRect> candidates;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        threshold(gray, binary, 120, 255, THRESH_BINARY);

        vector<vector<Point>> contours;
        findContours(binary, contours,
                     RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        Mat output = frame.clone();

        // 提取当前帧亮角候选
        for (size_t i = 0; i < contours.size(); ++i) {
            double area = contourArea(contours[i]);

            if (area < 80.0)
                continue;

            RotatedRect rect = minAreaRect(contours[i]);

            float length = max(rect.size.width, rect.size.height);
            float width  = min(rect.size.width, rect.size.height);

            if (width <= 0.0f)
                continue;

            float aspect = length / width;

            double rectangleArea =
                rect.size.width * rect.size.height;
            double fillRatio = area / rectangleArea;

            bool accepted =
                aspect <= 4.0f &&
                fillRatio >= 0.15 &&
                fillRatio <= 0.80;

            if (accepted)
                candidates.push_back(rect);

            // 单个区域：绿色通过，红色排除
            Scalar color = accepted
                ? Scalar(0, 255, 0)
                : Scalar(0, 0, 255);

            Point2f corners[4];
            rect.points(corners);

            for (int j = 0; j < 4; ++j) {
                Point p(cvRound(corners[j].x),
                        cvRound(corners[j].y));
                Point q(cvRound(corners[(j + 1) % 4].x),
                        cvRound(corners[(j + 1) % 4].y));

                line(output, p, q, color, 1);
            }
        }

        auto hypotheses =
            findMarkerHypotheses(candidates, binary);

        cout << "Frame " << frameIndex
             << ", candidates=" << candidates.size()
             << ", hypotheses=" << hypotheses.size()
             << '\n';

        // 整体候选：黄色连线，蓝色关键点
        for (size_t h = 0; h < hypotheses.size(); ++h) {
            const auto& points = hypotheses[h];

            cout << "  Hypothesis " << h << ": ";

            for (int i = 0; i < 4; ++i) {
                Point p(cvRound(points[i].x),
                        cvRound(points[i].y));
                Point q(cvRound(points[(i + 1) % 4].x),
                        cvRound(points[(i + 1) % 4].y));

                line(output, p, q, Scalar(0, 255, 255), 2);
                circle(output, p, 4,
                       Scalar(255, 0, 0), FILLED);

                putText(output, names[i], p + Point(5, -5),
                        FONT_HERSHEY_SIMPLEX, 0.5,
                        Scalar(0, 255, 255), 1);
            }

            cout << '\n';
        }

        // 当前帧是否有通过全部检查的整体候选
        bool detected = !hypotheses.empty();

        string status = detected ? "Detected" : "Not detected";

        putText(output, status,
                Point(20, 40),
                FONT_HERSHEY_SIMPLEX, 1.0,
                Scalar(0, 255, 255), 2);

        // 保存带识别标注的当前帧
        writer.write(output);

        imshow("Marker hypotheses", output);
        imshow("Binary", binary);

        int key = waitKey(10);

        if (key == 27)
            break;

        if (key == 's' || key == 'S') {
            string prefix = "integration_" + to_string(frameIndex);

            imwrite(prefix + "_original.png", frame);
            imwrite(prefix + "_binary.png", binary);
            imwrite(prefix + "_result.png", output);

            cout << "Saved: " << prefix << '\n';
        }

        ++frameIndex;
    }

    return 0;
}