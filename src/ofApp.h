#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include "ofxOsc.h"
#include <thread>
#include <mutex>
#include <atomic>
#include "MovDetector.h"

#define SUPERCOLLIDER "127.0.0.1"
#define PORT 57120


class ofApp : public ofBaseApp{

	public:
		void setup() override;
		void update() override;
		void draw() override;
		void exit() override;

		void sendOscToSC(std::vector<glm::vec2> keypoints);

		void keyPressed(int key) override;
		void keyReleased(int key) override;
		void mouseMoved(int x, int y ) override;
		void mouseDragged(int x, int y, int button) override;
		void mousePressed(int x, int y, int button) override;
		void mouseReleased(int x, int y, int button) override;
		void mouseScrolled(int x, int y, float scrollX, float scrollY) override;
		void mouseEntered(int x, int y) override;
		void mouseExited(int x, int y) override;
		void windowResized(int w, int h) override;
		void dragEvent(ofDragInfo dragInfo) override;
		void gotMessage(ofMessage msg) override;
		
	private:
		void processFrames();

		MovDetector dancer;
		ofVideoGrabber vidGrabber;
		ofxOscSender sender;

		// Threading para o MovDetector
        std::thread workerThread;
        cv::Mat frameToProcess;
        std::mutex frameMutex;
        std::atomic<bool> newFrameAvailable{false};
        std::atomic<bool> stopThread{false};

		int width = 640;
		int height = 480;
		
};
