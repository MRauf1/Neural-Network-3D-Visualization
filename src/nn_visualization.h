#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <random>
#include <fstream>
#include <experimental/filesystem>
#include <regex>
#include "ofMain.h"
#include "nn.h"

using Eigen::MatrixXd;


// NN for Neural Network
class NNVisualization {

    private:

        std::vector<MatrixXd> images;
        std::vector<int> labels;
        std::vector<std::string> image_paths;
        NN nn;
        int index = 0;
        ofImage image;
        int kImageSize = 32;


    public:

        NNVisualization();

        void Initialize(std::vector<std::string> image_paths, std::vector<MatrixXd> images, std::vector<int> labels, NN nn);

        void DrawImage();

};
