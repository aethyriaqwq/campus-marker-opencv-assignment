#include "detector.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs=std::filesystem;
void ensureParent(const std::string& path) {
    if(!path.empty() && fs::path(path).has_parent_path()) fs::create_directories(fs::path(path).parent_path());
}
int main(int argc,char** argv) {
    try {
        std::string input,output,csv,snapshots;
        int camera=-1,maxFrames=0,start=0,snapshotStart=0,snapshotCount=6;
        bool show=false;
        marker::Config config;
        for(int i=1;i<argc;++i) {
            std::string key=argv[i];
            auto value=[&](){if(i+1>=argc) throw std::runtime_error("Missing value for "+key); return std::string(argv[++i]);};
            if(key=="--input") input=value();
            else if(key=="--camera") camera=std::stoi(value());
            else if(key=="--output") output=value();
            else if(key=="--csv") csv=value();
            else if(key=="--snapshots") snapshots=value();
            else if(key=="--snapshot-start") snapshotStart=std::stoi(value());
            else if(key=="--snapshot-count") snapshotCount=std::stoi(value());
            else if(key=="--max-frames") maxFrames=std::stoi(value());
            else if(key=="--start") start=std::stoi(value());
            else if(key=="--width") config.workWidth=std::stoi(value());
            else if(key=="--threshold") config.threshold=std::stoi(value());
            else if(key=="--min-score") config.minScore=std::stof(value());
            else if(key=="--alpha") config.smoothing=std::stof(value());
            else if(key=="--show") show=true;
            else if(key=="--help") {
                std::cout<<"Usage: marker (--input VIDEO | --camera N) [--output demo.avi] [--csv frames.csv]\n"
                         <<"  --show                 Display; Esc/q to quit (default headless)\n"
                         <<"  --width 960            Detection width; 0 = original. Output always original size\n"
                         <<"  --threshold 180        Brightness segmentation threshold (1..254)\n"
                         <<"  --min-score 0.64       Pattern F1 threshold (0..1)\n"
                         <<"  --alpha 0.75           Current frame smoothing weight (0..1]\n"
                         <<"  --start 0 --max-frames 0  Video starting frame; 0 max means until EOF\n"
                         <<"  --snapshots DIR --snapshot-start 0 --snapshot-count 6\n";
                return 0;
            } else throw std::runtime_error("Unknown argument: "+key);
        }
        if((input.empty() && camera<0) || (!input.empty() && camera>=0)) throw std::runtime_error("Select exactly one of --input or --camera; see --help");
        if(start<0 || maxFrames<0 || snapshotStart<0 || snapshotCount<0) throw std::runtime_error("Frame indices/counts must be nonnegative");
        if(camera>=0 && start!=0) throw std::runtime_error("--start is for video files only");
        if(!input.empty()) {
            for(const auto& destination:{output,csv})
                if(!destination.empty() && fs::weakly_canonical(input)==fs::weakly_canonical(destination)) throw std::runtime_error("Output must not overwrite input");
        }
        if(!output.empty() && fs::path(output).extension()!=".avi") throw std::runtime_error("Use .avi output (MJPG); see README for evidence");
        if(!output.empty() && !csv.empty() && fs::weakly_canonical(output)==fs::weakly_canonical(csv)) throw std::runtime_error("Video and CSV output paths must differ");
        marker::Detector detector(config);
        cv::VideoCapture capture;
        if(camera>=0) capture.open(camera); else capture.open(input);
        if(!capture.isOpened()) throw std::runtime_error("Could not open input");
        if(start>0 && !capture.set(cv::CAP_PROP_POS_FRAMES,start)) throw std::runtime_error("Could not seek to start frame");
        double fps=capture.get(cv::CAP_PROP_FPS);
        if(!std::isfinite(fps) || fps<=0 || fps>1000) {fps=30; std::cerr<<"Warning: invalid input FPS; output fallback is 30 FPS\n";}
        std::ofstream rows;
        if(!csv.empty()) {
            ensureParent(csv); rows.open(csv); if(!rows) throw std::runtime_error("Could not open CSV");
            rows<<"frame,timestamp_ms,detected,id,score,bbox_x,bbox_y,bbox_w,bbox_h,lt_x,lt_y,rt_x,rt_y,rb_x,rb_y,lb_x,lb_y\n"<<std::fixed<<std::setprecision(3);
        }
        if(!snapshots.empty()) fs::create_directories(snapshots);
        cv::VideoWriter writer;
        cv::Mat frame;
        int count=0,positive=0,totalDetections=0;
        double detectMs=0;
        auto began=std::chrono::steady_clock::now();
        while((maxFrames==0 || count<maxFrames) && capture.read(frame)) {
            if(frame.empty()) break;
            int index=start+count;
            auto t=std::chrono::steady_clock::now();
            auto detections=detector.process(frame);
            detectMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
            if(!detections.empty()) ++positive;
            totalDetections+=static_cast<int>(detections.size());
            double timestamp=camera>=0?std::chrono::duration<double,std::milli>(t-began).count():1000.0*index/fps;
            if(rows.is_open()) {
                if(detections.empty()) rows<<index<<','<<timestamp<<",0,-1,0,,,,,,,,,,,,\n";
                for(const auto& d:detections) {
                    float x0=1e9,y0=1e9,x1=0,y1=0;
                    for(auto p:d.corners) {x0=std::min(x0,p.x); y0=std::min(y0,p.y); x1=std::max(x1,p.x); y1=std::max(y1,p.y);}
                    rows<<index<<','<<timestamp<<",1,"<<d.id<<','<<d.score<<','<<x0<<','<<y0<<','<<x1-x0<<','<<y1-y0;
                    for(auto p:d.corners) rows<<','<<p.x<<','<<p.y;
                    rows<<'\n';
                }
                if(!rows) throw std::runtime_error("Failed writing CSV");
            }
            marker::draw(frame,detections,index);
            if(!output.empty()) {
                if(!writer.isOpened()) {
                    ensureParent(output);
                    writer.open(output,cv::VideoWriter::fourcc('M','J','P','G'),fps,frame.size());
                    if(!writer.isOpened()) throw std::runtime_error("Could not open MJPG output writer");
                }
                writer.write(frame);
            }
            if(!snapshots.empty() && index>=snapshotStart && index<snapshotStart+snapshotCount)
                if(!cv::imwrite((fs::path(snapshots)/cv::format("frame_%06d.jpg",index)).string(),frame,{cv::IMWRITE_JPEG_QUALITY,90})) throw std::runtime_error("Failed writing screenshot");
            ++count;
            if(show) {cv::imshow("Campus marker",frame); int key=cv::waitKey(1); if(key==27 || key=='q') break;}
        }
        if(count==0) throw std::runtime_error("No frames decoded; check input/start");
        writer.release();
        std::cout<<"frames="<<count<<" positive_frames="<<positive<<" detections="<<totalDetections
                 <<" input_fps="<<fps<<" mean_detection_ms="<<detectMs/count<<"\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"Error: "<<e.what()<<'\n'; return 1;}
}
