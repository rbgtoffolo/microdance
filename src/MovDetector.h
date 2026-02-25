#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include <opencv2/dnn.hpp>
#include <mutex>

class MovDetector {
    public:
        void setup(string modelPath);
        void update(cv::Mat frame);
        void trace();

        std::vector<glm::vec2> getPoseKeypoints();
        
    private:
        // Variáveis para processamento com OpenCV moderno (via ofxCv)
        cv::Mat currentFrame;
        cv::Mat previousFrame;
        cv::Mat diffFrame;
        
        // Pose detection
        cv::dnn::Net net;
        std::vector<glm::vec2> keypoints;
        std::mutex mtx;

};
