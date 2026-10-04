#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>

#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

cv::Mat frame_annotation(cv::Mat frame){
    cv::Mat origin = frame.clone();
    //存储原帧

    //1.去噪
    cv::Mat blur;
    cv::medianBlur(origin, blur, 3);

    //2.二值化(转成灰度图效果更好？)+检测是否有目标
    cv::Mat gray, binary;
    cv::cvtColor(blur, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 200, 255, cv::THRESH_BINARY);//经验参数
    // cv::imshow("binary", binary);

    //3.形态学处理：把不连续灯条连起来
    cv::Mat process1, process2;
    cv::Mat progress;
    const cv::Mat kernel_vertical = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(5, 60));
    const cv::Mat kernel_horizontal = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(60, 5));
    const cv::Mat kernel_erode = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(9, 9));
    cv::erode(binary, process1, kernel_erode);
    cv::dilate(process1, process2, kernel_vertical);
    cv::dilate(process2, progress, kernel_horizontal);

    //4.轮廓提取+判断是否出现目标
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(progress, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //5.画旋转矩形
    cv::Mat rotated_rect = frame.clone();
    if (contours.size()==0 || contours.size() > 1) {
        cv::putText(rotated_rect, "target not exists", cv::Point{50, 50},
                    cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar{0, 0, 255}, 4, 
                    cv::LINE_AA);
        return rotated_rect;
    }
    std::vector<cv::Point> contour = contours[0];
    if (contours.size() == 1 && cv::contourArea(contour) < 30000) {
        cv::putText(rotated_rect, "target not exists", cv::Point{50, 50},
                    cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar{0, 0, 255}, 4, 
                    cv::LINE_AA);
        return rotated_rect;
    }
    
    cv::Point2f vertices[4];
    const cv::RotatedRect& rectangle = cv::minAreaRect(contour);
    rectangle.points(vertices);
    for (int i = 0; i < 4; i++) {
        cv::line(rotated_rect, vertices[i], vertices[(i + 1) % 4], 
                 cv::Scalar{255, 255, 0}, 1, cv::LINE_AA);
    }

    //6.标四角点+标"target"文字
    for (int i = 0; i < 4; i++) {
        cv::circle(rotated_rect, vertices[i], 5, cv::Scalar{0, 0, 255}, -1);
    }
    cv::putText(rotated_rect, "target exists", cv::Point{50, 50}, 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar{0, 255, 0}, 2,
                cv::LINE_AA);

    return rotated_rect;
}

int main(int argc, char** argv){
    if (argc != 3){
        std::cerr << "用法" << argv[0] << "<输入视频><输出目录>\n";
        return 1;
    }
    try{
        const fs::path input_path(argv[1]);//input_path = marker_video.avi
        const fs::path output_directory(argv[2]);// = results
        if (!fs::is_regular_file(input_path)){
            throw std::runtime_error("输入视频不存在: " + input_path.string());
        }
        cv::VideoCapture capture(input_path.string());
        fs::create_directories(output_directory.string());
        const fs::path video_path = output_directory / ("marker_annotated.mp4");

        double fps = capture.get(cv::CAP_PROP_FPS);
        if (!std::isfinite(fps) || fps <= 0.0) {
            fps = 30.0;
        }

        cv::VideoWriter result;
        cv::Mat frame;
        // capture.read(frame);
        while (capture.read(frame)){
            if (frame.empty()) {
                throw std::runtime_error("该视频出现空帧");
            }
            cv::Mat temp;//接收每一帧的处理结果
            temp = frame_annotation(frame);
            if (!result.isOpened()) {
                result.open(video_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                fps, frame.size(), true);
            }
            result.write(temp);
        }
        result.release();
    }catch (const std::exception& error){
        std::cerr << "错误: " << error.what() << '\n';
    }
}