#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
const int inf=1e9;
const int min_color=150,match_eps=20,ignore_area=25;
const double eps_ratio=0.15;
int eps;

struct Lightbar{
    int type;   //1-4分别为左上、右上、右下、左下
    cv::Point corner_point;   //顶点，用于主要判断，非常准确
    cv::Point inner_point;   //内顶点，辅助判断
    std::vector<cv::Point> other_point;   //其他角点，单纯用于画图展示
};

struct Armor{
    std::vector<cv::Point> corner;
};

//判断两数是否大约相等
bool equal(int a,int b){
    return std::abs(a-b)<=eps;
}

bool equal_point(cv::Point a,cv::Point b){
    return equal(a.x,b.x)&&equal(a.y,b.y);
}

int calc_dis(cv::Point a,cv::Point b){
    return (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y);
}

cv::Point getmid(const std::vector<cv::Point>& v){
    return v[v.size()>>1];
}
//我们只处理四个角处的lightbar，不管那两个正方形
//返回值的type为-1表明这不是我们要得lightbar
Lightbar processLightBar(std::vector<cv::Point> counters){
    Lightbar result; 
    result.type=-1;
    int min_x=inf,max_x=0,min_y=inf,max_y=0;
    for(cv::Point p:counters){
        max_x=std::max(max_x,p.x);
        min_x=std::min(min_x,p.x);
        max_y=std::max(max_y,p.y);
        min_y=std::min(min_y,p.y);
    }
    eps=std::ceil(1.0*(max_x+max_y-min_x-min_y)/2*eps_ratio);
    //获取lightbar外侧的角点
    std::vector<cv::Point> outer_keypoint;
    for(cv::Point p:counters)
        if((equal(p.x,min_x)||equal(p.x,max_x))&&
           (equal(p.y,min_y)||equal(p.y,max_y)))
            outer_keypoint.push_back(p);
    
    std::sort(outer_keypoint.begin(),outer_keypoint.end(),
              [](cv::Point a,cv::Point b)
              {return equal(a.x,b.x)?a.y<b.y:a.x<b.x;});
    //进行角点的去重
    std::vector<std::vector<cv::Point>> same_point;
    std::vector<cv::Point> temp;
    if(!outer_keypoint.size()) return result;
    cv::Point last_keypoint=outer_keypoint[0];
    for(cv::Point p:outer_keypoint){
        if(!equal_point(p,last_keypoint)){
            same_point.push_back(temp);
            temp.clear();
            last_keypoint=p;
        }        
        temp.push_back(p);
    }
    same_point.push_back(temp);
    //这步可以过滤掉正方形
    if(same_point.size()!=3) return result;
    //确定哪个是整个目标的顶点
    int corner_index=-1;
    for(int i=0;i<3;i++){
        cv::Point This=same_point[i][0];
        cv::Point other1=same_point[(i+1)%3][0];
        cv::Point other2=same_point[(i+2)%3][0];
        if((equal(This.x,other1.x)||equal(This.y,other1.y))&&
           (equal(This.x,other2.x)||equal(This.y,other2.y))){
            if(corner_index!=-1) return result;
            corner_index=i;
        }
    }
    if(corner_index==-1) return result;
    //判断这个lightbar在哪个角上
    cv::Point aim_point=getmid(same_point[corner_index]);   //外顶点    
    //inner即“L”型lightbar的外接矩形靠里的两条边,outer同理
    //"L"型inner与outer的两角点
    cv::Point outer_sidepoint1=getmid(same_point[(corner_index+1)%3]);
    cv::Point outer_sidepoint2=getmid(same_point[(corner_index+2)%3]);
    if(equal(aim_point.x,min_x)&&equal(aim_point.y,min_y)) result.type=1;
    if(equal(aim_point.x,max_x)&&equal(aim_point.y,min_y)) result.type=2;
    if(equal(aim_point.x,max_x)&&equal(aim_point.y,max_y)) result.type=3;
    if(equal(aim_point.x,min_x)&&equal(aim_point.y,max_y)) result.type=4;
    //鉴于same_point有很多个，我们整出顶点的准确位置
    int precise_x=aim_point.x;
    int precise_y=aim_point.y;
    int inner_x,inner_y;
    for(cv::Point p:same_point[corner_index]){
        if(result.type==1||result.type==4)
            precise_x=std::min(precise_x,p.x),inner_x=max_x;
        else precise_x=std::max(precise_x,p.x),inner_x=min_x;
        if(result.type==1||result.type==2)
            precise_y=std::min(precise_y,p.y),inner_y=max_y;
        else precise_y=std::max(precise_y,p.y),inner_y=min_y;
    }
    result.corner_point.x=precise_x;
    result.corner_point.y=precise_y;
    if(!equal(outer_sidepoint1.x,inner_x))
        std::swap(outer_sidepoint1,outer_sidepoint2);
    result.other_point.push_back(outer_sidepoint1);
    result.other_point.push_back(outer_sidepoint2);
    //已经确定了外角点，我们获取内角点
    //两条inner边上的另一个角点
    cv::Point inner_sidepoint1={-1,-1},inner_sidepoint2={-1,-1};
    //因为我们已知inner边上的角点分别是inner边的两端
    cv::Point min_sidepoint=outer_sidepoint1,max_sidepoint=outer_sidepoint1;
    for(cv::Point p:counters)
        if(equal(p.x,inner_x)){
            if(p.y<min_sidepoint.y) min_sidepoint=p;
            if(p.y>max_sidepoint.y) max_sidepoint=p;
        }
    if(abs(min_sidepoint.y-outer_sidepoint1.y)<
       abs(max_sidepoint.y-outer_sidepoint1.y))
        result.other_point.push_back(inner_sidepoint1=max_sidepoint);
    else result.other_point.push_back(inner_sidepoint1=min_sidepoint);

    min_sidepoint=outer_sidepoint2,max_sidepoint=outer_sidepoint2;
    for(cv::Point p:counters)
        if(equal(p.y,inner_y)){
            if(p.x<min_sidepoint.x) min_sidepoint=p;
            if(p.x>max_sidepoint.x) max_sidepoint=p;
        }
    if(abs(min_sidepoint.x-outer_sidepoint2.x)<
       abs(max_sidepoint.x-outer_sidepoint2.x))
        result.other_point.push_back(inner_sidepoint2=max_sidepoint);
    else result.other_point.push_back(inner_sidepoint2=min_sidepoint);
    //找内顶点
    cv::Point inner_keypoint={-10000,-10000};
    cv::Point rough_position={inner_sidepoint2.x,inner_sidepoint1.y};
    for(cv::Point p:counters)
        if(equal_point(p,rough_position)&&
           calc_dis(p,rough_position)<=
           calc_dis(inner_keypoint,rough_position)) 
            inner_keypoint=p;
    result.inner_point=inner_keypoint;
    //整体判断
    cv::Point all_point[6]={result.corner_point,outer_sidepoint1,
                            inner_sidepoint1,inner_keypoint,
                            inner_sidepoint2,outer_sidepoint2};      
    for(int i=0;i<6;i++){
        cv::Point This=all_point[i];
        cv::Point Next=all_point[(i+1)%6];
        if(!(i%2==0&&equal(This.y,Next.y))&&
           !(i%2==1&&equal(This.x,Next.x))){
            result.type=-1;
            return result;
        }
    }
    return result;
}

bool isRect(cv::Point a,cv::Point b,cv::Point c,cv::Point d){
    return equal(a.y,b.y)&&equal(b.x,c.x)&&equal(c.y,d.y)&&equal(d.x,a.x);
}

Armor matchLightbar(const std::vector<Lightbar>& lightbar){
    eps=match_eps;
    Armor result;
    std::vector<std::vector<Lightbar>> type_lightbar(5);
    for(Lightbar l:lightbar){
        if(l.type<1||l.type>4) continue;
        type_lightbar[l.type].push_back(l);
    }
    for(Lightbar typ1:type_lightbar[1])
    for(Lightbar typ2:type_lightbar[2])
    for(Lightbar typ3:type_lightbar[3])
    for(Lightbar typ4:type_lightbar[4]){
        if(isRect(typ1.corner_point,typ2.corner_point,typ3.corner_point,typ4.corner_point)&&
           isRect(typ1.inner_point,typ2.inner_point,typ3.inner_point,typ4.inner_point)){
                result.corner.push_back(typ1.corner_point);
                result.corner.push_back(typ2.corner_point);
                result.corner.push_back(typ3.corner_point);
                result.corner.push_back(typ4.corner_point);
                return result;
           }
    }
    return result;
}

cv::Mat getCounter(const cv::Mat frame,std::vector<std::vector<cv::Point>>& counters){
    cv::Mat result;
    //1.滤波
    cv::Mat denoise_image;
    cv::medianBlur(frame,denoise_image,3);
    //2.二值化
    cv::Mat binary_image;
    cv::inRange(denoise_image,cv::Scalar(min_color,min_color,min_color),
                cv::Scalar(255,255,255),binary_image);
    //3.形态学处理
    const cv::Mat close_kernel=
        cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    cv::Mat close_dilated_image;
    cv::dilate(binary_image,close_dilated_image,close_kernel);
    cv::Mat close_image;
    cv::erode(close_dilated_image,close_image,close_kernel);
    const cv::Mat open_kernal=
        cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    cv::Mat open_eroded_image;
    cv::erode(close_image,open_eroded_image,open_kernal);
    cv::Mat morphology_image;
    cv::dilate(open_eroded_image,morphology_image,open_kernal);
    //4.轮廓提取
    cv::findContours(morphology_image,counters,
                     cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    return morphology_image;
}

void drawDot(cv::Mat image,cv::Point p){
    cv::circle(image,cv::Point(p.x,p.y),5,cv::Scalar(255,0,0),cv::FILLED);
}

int main(int argc,char** argv){
    if(argc!=3){
        std::cout<<"输入格式错误，应为 ："<<argv[0]<<" <输入视频.mp4> <输出目录>\n";
        return -1;
    }
    const fs::path input_path(argv[1]);
    const fs::path output_path(argv[2]);
    if(!fs::is_regular_file(input_path)){
        std::cout<<"未找到输入文件\n";
        return -1;
    }
    cv::VideoCapture capture(input_path.string());
    if(!capture.isOpened()){
        std::cout<<"输出路径错误\n";
        return -1;
    }
    cv::VideoWriter writer;
    double fps=capture.get(cv::CAP_PROP_FPS);
    if(!std::isfinite(fps)||fps <= 0.0) fps=30;
    cv::Mat image;
    while(capture.read(image)){
        if(!writer.isOpened())
            writer.open(output_path.string(),cv::VideoWriter::fourcc('m','p','4','v'),
                        fps,image.size(),1);
        std::vector<std::vector<cv::Point>> counters;
        cv::Mat process=getCounter(image,counters);
        std::vector<Lightbar> all_lightbar;
        for(std::vector<cv::Point> v:counters){
            if(cv::contourArea(v)<=ignore_area) continue;
            Lightbar This=processLightBar(v);
            if(1<=This.type&&This.type<=4) all_lightbar.push_back(This);
        }
        for(Lightbar l:all_lightbar){
            drawDot(image,l.corner_point);
            drawDot(image,l.inner_point);
            for(cv::Point p:l.other_point) drawDot(image,p);
        }
        Armor result=matchLightbar(all_lightbar);        
        if(result.corner.size()==4)
            for(int i=0;i<4;i++)
                cv::line(image,result.corner[i],result.corner[(i+1)%4],
                         cv::Scalar(0,0,255),2,cv::LINE_AA);
        writer.write(image);
    }
    writer.release();
    return 0;
}