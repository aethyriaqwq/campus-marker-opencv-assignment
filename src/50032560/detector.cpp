#include "detector.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace marker {
namespace {
float cross(cv::Point2f a, cv::Point2f b) { return a.x*b.y-a.y*b.x; }
cv::Point2f center(const Quad& q) { return (q[0]+q[1]+q[2]+q[3])*0.25f; }
float side(const Quad& q) { return static_cast<float>(cv::norm(q[1]-q[0])+cv::norm(q[2]-q[1])+cv::norm(q[3]-q[2])+cv::norm(q[0]-q[3]))/4; }
float area(const Quad& q) { return static_cast<float>(std::abs(cv::contourArea(std::vector<cv::Point2f>(q.begin(),q.end())))); }
bool valid(const Quad& q, cv::Size size) {
    if (!cv::isContourConvex(std::vector<cv::Point2f>(q.begin(),q.end())) || area(q)<500) return false;
    float shortest=1e9f, longest=0;
    for(int i=0;i<4;++i) {
        if(q[i].x<1 || q[i].y<1 || q[i].x>=size.width-1 || q[i].y>=size.height-1) return false;
        float a=static_cast<float>(cv::norm(q[(i+1)%4]-q[i]));
        shortest=std::min(shortest,a); longest=std::max(longest,a);
        auto u=q[(i+1)%4]-q[i], v=q[(i+3)%4]-q[i];
        if(std::abs(u.dot(v)/(cv::norm(u)*cv::norm(v)))>0.7) return false;
    }
    return shortest>20 && longest/shortest<2.3;
}
float scoreQuad(const cv::Mat& mask, const cv::Mat& model, const Quad& q) {
    Quad dest{{{0,0},{159,0},{159,159},{0,159}}};
    cv::Mat patch;
    cv::warpPerspective(mask,patch,cv::getPerspectiveTransform(q.data(),dest.data()),{160,160},cv::INTER_NEAREST);
    // Only a current-frame image can validate a candidate; no tracking-only output.
    if(cv::countNonZero(patch(cv::Rect(55,55,50,50)))>250) return 0;
    float best=0;
    for(int rotation=0;rotation<4;++rotation) {
        cv::Mat overlap;
        cv::bitwise_and(patch,model,overlap);
        float tp=static_cast<float>(cv::countNonZero(overlap));
        float f1=2*tp/static_cast<float>(cv::countNonZero(patch)+cv::countNonZero(model));
        // Require evidence in all four corners, even the fragmented corner.
        bool all=true;
        for(auto r : {cv::Rect(0,0,60,60),cv::Rect(100,0,60,60),cv::Rect(100,100,60,60),cv::Rect(0,100,60,60)})
            if(cv::countNonZero(overlap(r))<0.42*cv::countNonZero(model(r))) all=false;
        if(all) best=std::max(best,f1);
        cv::rotate(patch,patch,cv::ROTATE_90_CLOCKWISE);
    }
    return best;
}
// Fit each outside edge from bright contour points close to the predicted edge.
// Intersections of independently fitted lines preserve perspective deformation.
Quad refine(const Quad& q, const std::vector<cv::Point>& points) {
    std::array<cv::Vec4f,4> lines;
    float s=side(q);
    for(int i=0;i<4;++i) {
        auto a=q[i], b=q[(i+1)%4], v=b-a;
        float length=static_cast<float>(cv::norm(v)); v*=1/length;
        std::vector<cv::Point2f> near;
        // Select outermost pixels in small bins; avoids fitting the inner lamp edge.
        constexpr int bins=24;
        std::array<float,bins> outward; outward.fill(1e9f);
        std::array<cv::Point2f,bins> selected{};
        for(auto p:points) {
            cv::Point2f d=cv::Point2f(p)-a;
            float t=d.dot(v), dist=cross(v,d);
            if(t< -0.025f*length || t>1.025f*length || std::abs(dist)>0.12f*s) continue;
            int k=std::clamp(static_cast<int>(t/length*bins),0,bins-1);
            if(dist<outward[k]) { outward[k]=dist; selected[k]=p; }
        }
        for(int k=0;k<bins;++k) if(outward[k]<1e8f) near.push_back(selected[k]);
        if(near.size()<6) return q;
        cv::fitLine(near,lines[i],cv::DIST_HUBER,0,0.01,0.01);
    }
    Quad result;
    for(int i=0;i<4;++i) {
        auto l=lines[(i+3)%4], m=lines[i];
        cv::Point2f a(l[2],l[3]), u(l[0],l[1]), b(m[2],m[3]), v(m[0],m[1]);
        float den=cross(u,v);
        if(std::abs(den)<0.2) return q;
        result[i]=a+u*(cross(b-a,v)/den);
        if(cv::norm(result[i]-q[i])>s*0.22) return q;
    }
    return result;
}
} // namespace

cv::Mat makeTemplate(int size) {
    cv::Mat m(160,160,CV_8U,cv::Scalar(0));
    auto rect=[&](int x,int y,int w,int h){cv::rectangle(m,{x,y,w,h},255,cv::FILLED);};
    rect(0,0,60,16); rect(0,0,16,60);
    rect(0,144,60,16); rect(0,100,16,60);
    rect(100,144,60,16); rect(144,100,16,60);
    rect(100,0,16,16); rect(132,0,28,16); rect(144,0,16,28); rect(144,44,16,16);
    if(size!=160) cv::resize(m,m,{size,size},0,0,cv::INTER_NEAREST);
    return m;
}
Quad orderCorners(Quad q) {
    auto c=center(q);
    std::sort(q.begin(),q.end(),[&](auto a,auto b){return std::atan2(a.y-c.y,a.x-c.x)<std::atan2(b.y-c.y,b.x-c.x);});
    // LT is the smallest x+y; tie is the smaller y (image-relative convention).
    auto first=std::min_element(q.begin(),q.end(),[](auto a,auto b){return std::abs(a.x+a.y-b.x-b.y)<1e-3 ? a.y<b.y : a.x+a.y<b.x+b.y;});
    std::rotate(q.begin(),first,q.end());
    return q;
}
Detector::Detector(Config config):config_(config),template_(makeTemplate()) {
    if(config.workWidth<0 || config.threshold<1 || config.threshold>254 || config.minScore<=0 || config.minScore>1 || config.smoothing<=0 || config.smoothing>1)
        throw std::invalid_argument("Invalid detector configuration");
}
void Detector::reset() { previous_.clear(); lastSize_={}; nextId_=0; }
std::vector<Detection> Detector::process(const cv::Mat& frame) {
    if(frame.empty() || frame.type()!=CV_8UC3) throw std::invalid_argument("Expected nonempty CV_8UC3 BGR frame");
    if(lastSize_!=frame.size()) { previous_.clear(); lastSize_=frame.size(); }
    cv::Mat work,gray,mask;
    double scale=config_.workWidth>0 ? std::min(1.0,double(config_.workWidth)/frame.cols) : 1.0;
    cv::resize(frame,work,{cvRound(frame.cols*scale),cvRound(frame.rows*scale)},0,0,cv::INTER_AREA);
    const float sx=float(frame.cols)/work.cols, sy=float(frame.rows)/work.rows;
    cv::cvtColor(work,gray,cv::COLOR_BGR2GRAY);
    cv::threshold(gray,mask,config_.threshold,255,cv::THRESH_BINARY);
    cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_RECT,{3,3}));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
    std::vector<cv::Point> support;
    for(const auto& c:contours) if(cv::contourArea(c)>5) support.insert(support.end(),c.begin(),c.end());
    std::vector<Detection> candidates;
    for(const auto& c:contours) {
        double a=cv::contourArea(c);
        if(a<35 || a>mask.total()*0.06) continue;
        auto r=cv::minAreaRect(c);
        float small=std::min(r.size.width,r.size.height), big=std::max(r.size.width,r.size.height);
        if(small<9 || big/small>2 || a/r.size.area()<0.22 || a/r.size.area()>0.82) continue;
        Quad box; r.points(box.data()); box=orderCorners(box);
        // Every full L offers an independent marker hypothesis; works with multiple targets.
        for(int k=0;k<4;++k) {
            auto origin=box[k], u=box[(k+1)%4]-origin, v=box[(k+3)%4]-origin;
            Quad q=orderCorners({origin,origin+u*(80.f/30),origin+(u+v)*(80.f/30),origin+v*(80.f/30)});
            if(!valid(q,mask.size())) continue;
            q=refine(q,support);
            if(!valid(q,mask.size())) continue;
            float score=scoreQuad(mask,template_,q);
            if(score>=config_.minScore) candidates.push_back({q,score,-1});
        }
    }
    std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.score>b.score;});
    std::vector<Detection> found;
    for(auto d:candidates) {
        bool duplicate=false;
        for(const auto& kept:found) {
            // Polygon clipping can be numerically unstable for near-coincident edges.
            // A center/scale gate catches the same marker before that calculation.
            float ratio=side(d.corners)/side(kept.corners);
            if(ratio>0.7f && ratio<1.43f && cv::norm(center(d.corners)-center(kept.corners))<0.25f*std::min(side(d.corners),side(kept.corners))) {
                duplicate=true;
                break;
            }
            std::vector<cv::Point2f> intersection;
            float inter=cv::intersectConvexConvex(std::vector<cv::Point2f>(d.corners.begin(),d.corners.end()),std::vector<cv::Point2f>(kept.corners.begin(),kept.corners.end()),intersection);
            if(inter/std::min(area(d.corners),area(kept.corners))>0.55) duplicate=true;
        }
        if(!duplicate) found.push_back(d);
    }
    std::vector<bool> used(previous_.size(),false);
    for(auto& d:found) {
        for(auto& p:d.corners) { p.x*=sx; p.y*=sy; }
        float best=side(d.corners)*0.45f; int matched=-1;
        for(size_t j=0;j<previous_.size();++j) {
            float distance=static_cast<float>(cv::norm(center(d.corners)-center(previous_[j].corners)));
            float ratio=side(d.corners)/side(previous_[j].corners);
            if(!used[j] && ratio>0.65 && ratio<1.55 && distance<best) { best=distance; matched=static_cast<int>(j); }
        }
        if(matched>=0) {
            used[matched]=true; d.id=previous_[matched].id;
            // Smooth small jitter only. Large motion uses current measurement directly.
            float maxMove=0;
            for(int k=0;k<4;++k) maxMove=std::max(maxMove,static_cast<float>(cv::norm(d.corners[k]-previous_[matched].corners[k])));
            if(maxMove<side(d.corners)*0.08f)
                for(int k=0;k<4;++k) d.corners[k]=config_.smoothing*d.corners[k]+(1-config_.smoothing)*previous_[matched].corners[k];
        } else d.id=nextId_++;
    }
    previous_=found; // Empty frame clears tracks immediately; never reuse a missing detection.
    return found;
}
void draw(cv::Mat& frame,const std::vector<Detection>& detections,int frameIndex) {
    const cv::Scalar colors[]={{0,255,0},{0,220,255},{255,100,20},{255,0,255}};
    const char* names[]={"LT","RT","RB","LB"};
    for(const auto& d:detections) {
        std::vector<cv::Point> polygon;
        for(auto p:d.corners) polygon.emplace_back(cvRound(p.x),cvRound(p.y));
        cv::polylines(frame,polygon,true,{0,255,0},2,cv::LINE_AA);
        for(int k=0;k<4;++k) {
            cv::circle(frame,d.corners[k],5,colors[k],cv::FILLED,cv::LINE_AA);
            cv::putText(frame,names[k],d.corners[k]+cv::Point2f(6,k<2?-9:22),cv::FONT_HERSHEY_SIMPLEX,0.65,colors[k],2,cv::LINE_AA);
        }
        cv::putText(frame,cv::format("ID %d score %.2f",d.id,d.score),d.corners[0]+cv::Point2f(-10,-32),cv::FONT_HERSHEY_SIMPLEX,0.6,{0,255,0},2,cv::LINE_AA);
    }
    cv::rectangle(frame,{0,0,frame.cols,55},{20,20,20},cv::FILLED);
    std::string status=detections.empty()?"NOT DETECTED":"DETECTED: "+std::to_string(detections.size());
    cv::putText(frame,"50032560 | frame "+std::to_string(frameIndex)+" | "+status,{18,36},cv::FONT_HERSHEY_SIMPLEX,0.8,detections.empty()?cv::Scalar(0,160,255):cv::Scalar(0,255,0),2,cv::LINE_AA);
}
} // namespace marker
