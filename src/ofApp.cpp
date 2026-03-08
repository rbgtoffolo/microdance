#include "ofApp.h"
#include "ofxOsc.h"


//--------------------------------------------------------------
void ofApp::setup(){

    sender.setup(SUPERCOLLIDER, PORT);
    ofxOscMessage m;
    m.setAddress("/start");
    m.addIntArg(width);
    m.addIntArg(height);
    sender.sendMessage(m);
    
    // Pose Detector
    string modelPath = ofToDataPath("dnn/yolov8n-pose.onnx");
    dancer.setup(modelPath);
    
    //vidGrabber.setVerbose(true);
    vidGrabber.setup(640, 480);
    workerThread = std::thread(&ofApp::processFrames, this);
}

//--------------------------------------------------------------
void ofApp::update(){
    vidGrabber.update();
    
    if (vidGrabber.isFrameNew()) {
        // Só processa um novo frame se o anterior já foi consumido pela thread.
        if (!newFrameAvailable) {
            std::lock_guard<std::mutex> lock(frameMutex);

            // Cria um cv::Mat temporário usando os dados do ofPixels (RGB) e clona para a thread
            ofPixels & pixels = vidGrabber.getPixels();
            cv::Mat temp(pixels.getHeight(), pixels.getWidth(), CV_8UC3, pixels.getData());
            frameToProcess = temp.clone();
            newFrameAvailable = true;
        }
    }

    sendOscToSC(dancer.getPoseKeypoints());

}
//--------------------------------------------------------------
void ofApp::draw(){
    ofSetColor(255);
    vidGrabber.draw(0, 0); // Desenha a imagem da câmera no fundo
    dancer.trace();
}

//--------------------------------------------------------------
void ofApp::exit(){
    stopThread = true;
    if (workerThread.joinable()) {
        workerThread.join();
    }
    vidGrabber.close();
}

void ofApp::processFrames() {
    while (!stopThread) {
        if (newFrameAvailable) {
            cv::Mat localFrame;
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                std::swap(localFrame, frameToProcess);
                newFrameAvailable = false; // Marca que o frame foi "pego" pela thread.
            }
            dancer.update(localFrame);
        } else {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(1ms);
        }
    }
}

void ofApp::sendOscToSC(std::vector<glm::vec2> keypoints){
      ofxOscMessage m;
    m.setAddress("/poses");
    for (const auto& kp : keypoints) {
        m.addFloatArg(kp.x);
        m.addFloatArg(kp.y);
    }
    sender.sendMessage(m);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    if (key == ' '){
        ofxOscMessage m;
        m.setAddress("/reset");
        sender.sendMessage(m);
    }

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseScrolled(int x, int y, float scrollX, float scrollY){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
