#include "detector.hpp"
#include <iostream>
#include <stdexcept>

void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
cv::Mat scene(const marker::Quad& q,cv::Size size={900,700}) {
    cv::Mat image(size,CV_8UC3,cv::Scalar(28,28,28)),mask,model=marker::makeTemplate();
    marker::Quad src{{{0,0},{159,0},{159,159},{0,159}}};
    cv::warpPerspective(model,mask,cv::getPerspectiveTransform(src.data(),q.data()),size);
    image.setTo(cv::Scalar(255,255,255),mask);
    return image;
}
int main(int argc,char** argv) {
    try {
        marker::Quad q{{{240,150},{445,160},{435,365},{230,355}}};
        for(int width:{0,480,960}) {
            marker::Config config; config.workWidth=width;
            marker::Detector detector(config);
            auto d=detector.process(scene(q));
            require(d.size()==1,"Expected one perspective marker");
            for(int k=0;k<4;++k) require(cv::norm(d[0].corners[k]-q[k])<7,"Original-pixel corner coordinate mismatch");
            int id=d[0].id;
            auto moved=q; for(auto& p:moved) p+=cv::Point2f(4,2);
            auto next=detector.process(scene(moved));
            require(next.size()==1 && next[0].id==id,"Small motion must preserve track ID");
            require(detector.process(cv::Mat(700,900,CV_8UC3,cv::Scalar(28,28,28))).empty(),"Blank frame must immediately clear previous output");
            require(detector.process(scene(q)).size()==1,"Re-entry must be detected");
            auto occluded=scene(q); cv::rectangle(occluded,{400,110,100,110},{28,28,28},cv::FILLED);
            require(detector.process(occluded).empty(),"Missing corner should not hallucinate full target");
            require(detector.process(cv::Mat(700,900,CV_8UC3,cv::Scalar(255,255,255))).empty(),"White frame is not a marker");
        }
        marker::Detector detector;
        auto two=scene(q); auto q2=q; for(auto& p:q2) p.x+=350;
        cv::max(two,scene(q2),two);
        require(detector.process(two).size()==2,"Two separated targets must both be detected");
        auto distractor=cv::Mat(700,900,CV_8UC3,cv::Scalar(28,28,28));
        for(auto p:q) cv::rectangle(distractor,cv::Rect(cvRound(p.x),cvRound(p.y),25,25),{255,255,255},cv::FILLED);
        require(detector.process(distractor).empty(),"Four bright squares are not the marker");
        if(argc>1) {
            cv::VideoCapture capture(argv[1]);
            require(capture.isOpened(),"Cannot open repository regression video");
            for(int index:{0,619,900}) {
                require(capture.set(cv::CAP_PROP_POS_FRAMES,index),"Cannot seek regression frame");
                cv::Mat frame;
                require(capture.read(frame),"Cannot read regression frame");
                auto found=detector.process(frame);
                require(found.size()==(index==900?0u:1u),"Repository frame detection count (including duplicate suppression) failed");
            }
        }
        std::cout<<"PASS: perspective, scaling, corners, continuity, immediate disappearance, re-entry, occlusion, negatives, multiple targets\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n'; return 1;}
}
