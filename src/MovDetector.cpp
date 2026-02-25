#include "MovDetector.h"
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace cv::dnn;

void MovDetector::setup(string modelPath) {
    // Carrega o arquivo .onnx do YOLOv8/26
    net = readNetFromONNX(modelPath);
    
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        ofLogNotice("MovDetector") << "CUDA detectado! Usando GPU.";
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA); // FP32 para maior precisão nos keypoints
    } else {
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
}

void MovDetector::update(cv::Mat frame) {
    if (frame.empty() || net.empty()) return;

    // Redimensiona para 320x320 para fluidez máxima na GTX 1650
    // swapRB=true: Converte RGB (webcam) para o formato esperado pelo modelo
    Mat blob = blobFromImage(frame, 1.0/255.0, Size(320, 320), Scalar(0,0,0), true, false);
    net.setInput(blob);

    // O YOLO retorna um tensor [1, 56, 2100] (pode variar conforme a versão)
    Mat output = net.forward();
    
    // Reformata: Linhas viram detecções, colunas viram os dados [x, y, w, h, conf, kpts...]
    Mat detections = output.reshape(1, output.size[1]).t(); 

    std::vector<glm::vec2> nextKeypoints;

    int bestIdx = -1;
    float maxConf = 0.0;

    // 1. Encontra a melhor detecção (maior confiança) entre todas as candidatas
    for (int i = 0; i < detections.rows; i++) {
        float confidence = detections.at<float>(i, 4);
        if (confidence > maxConf) {
            maxConf = confidence;
            bestIdx = i;
        }
    }

    // 2. Processa apenas a melhor detecção se ela superar o limiar
    if (maxConf > 0.5 && bestIdx != -1) {
            int i = bestIdx;
            // Os keypoints começam no índice 5.
            // Cada ponto tem 3 valores: [x, y, conf_do_ponto]
            for (int k = 0; k < 17; k++) {
                int baseIdx = 5 + (k * 3);
                float x = detections.at<float>(i, baseIdx);
                float y = detections.at<float>(i, baseIdx + 1);
                float conf_pt = detections.at<float>(i, baseIdx + 2);

                if (conf_pt > 0.5) {
                    // Mapeia 0-320 de volta para a resolução da câmera
                    float mappedX = x * frame.cols / 320.0;
                    float mappedY = y * frame.rows / 320.0;
                    nextKeypoints.push_back(glm::vec2(mappedX, mappedY));
                } else {
                    nextKeypoints.push_back(glm::vec2(-1, -1));
                }
            }
    }

    std::lock_guard<std::mutex> lock(mtx);
    keypoints = nextKeypoints;
}

void MovDetector::trace() {
    std::lock_guard<std::mutex> lock(mtx);
    if(keypoints.empty()) return;

    ofPushStyle();
    
    // Mapeamento COCO (17 pontos) - O padrão do YOLO
    std::vector<std::pair<int, int>> pairs = {
        {0,1}, {0,2}, {1,3}, {2,4},              // Rosto
        {5,6}, {5,11}, {6,12}, {11,12},          // Tronco
        {5,7}, {7,9}, {6,8}, {8,10},             // Braços
        {11,13}, {13,15}, {12,14}, {14,16}       // Pernas
    };
    
    ofSetLineWidth(3);
    for(auto& p : pairs) {
        if(keypoints[p.first].x != -1 && keypoints[p.second].x != -1) {
            ofSetColor(0, 255, 0, 200);
            ofDrawLine(keypoints[p.first], keypoints[p.second]);
        }
    }

    for(const auto& kp : keypoints) {
        if(kp.x != -1) {
            ofSetColor(255, 0, 50);
            ofDrawCircle(kp, 5);
        }
    }
    ofPopStyle();
}

std::vector<glm::vec2> MovDetector::getPoseKeypoints() {
    std::lock_guard<std::mutex> lock(mtx);
    return keypoints;
}
