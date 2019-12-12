#pragma once

#include "ofMain.h"
#include "nn_visualization.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		ofEasyCam camera;
		int kCameraSpeed = 10;
		int frame_divider = 30;

		int kImageSize = 32;
		int kChannels = 3;
		int kNumPixels = kImageSize * kImageSize * kChannels;
		std::vector<int> kNeurons = {kNumPixels, 750, 375, 1};

		NNVisualization nn_visualization;

};
