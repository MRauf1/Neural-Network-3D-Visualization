#include "ofApp.h"
#include "nndata.h"
#include "nn.h"

//--------------------------------------------------------------
void ofApp::setup(){

    NNData data;

    ofEnableDepthTest();
    ofSetVerticalSync(true);

    camera.setPosition(0, 0, 100);

    test.load("data/test/cat.10000.jpg");

}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

    camera.begin();

    /*
    ofFill();
    ofSetColor(255, 0, 0, 255);
    ofDrawRectangle(0, 0, 0, 30, 30);
    */

    test.draw(0, 0, 32, 32);

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
