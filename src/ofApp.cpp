#include "ofApp.h"
#include "nndata.h"
#include "nn.h"

//--------------------------------------------------------------
void ofApp::setup(){

    std::cout << "Started preprocessing data" << std::endl;

    NNData data;
    std::pair<std::vector<Mat>, std::vector<int>> validation_data = data.LoadFromDirectory("../bin/data/validation");
    std::vector<Mat> reshaped_data = data.ConvertTo1D(validation_data.first);
    std::vector<MatrixXd> images = data.ConvertToEigen(reshaped_data);
    images = data.Preprocess(images);
    std::vector<std::string> image_paths = data.GetImagePaths();

    std::cout << "Finished preprocessing data" << std::endl;
    std::cout << "Started loading the model" << std::endl;

    NN nn(kNeurons, 0.01);
    nn.LoadModel();

    std::cout << "Finished loading the model" << std::endl;

    nn_visualization.Initialize(image_paths, images, validation_data.second, nn);

    ofEnableDepthTest();
    ofSetVerticalSync(true);

    camera.setPosition(500, 500, 1500);


}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

    camera.begin();

    nn_visualization.PassFeedForward();
    nn_visualization.DrawImage();
    nn_visualization.DrawAllLayers();

    camera.end();

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

    // Camera movement keys
    if(key == 'w') {
        camera.move(0, kCameraSpeed, 0);
    } else if(key == 'a') {
        camera.move(-kCameraSpeed, 0, 0);
    } else if(key == 's') {
        camera.move(0, -kCameraSpeed, 0);
    } else if(key == 'd') {
        camera.move(kCameraSpeed, 0, 0);
    } else if(key == 'q') {
        camera.move(0, 0, -kCameraSpeed);
    } else if(key == 'e') {
        camera.move(0, 0, kCameraSpeed);
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
