#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <experimental/filesystem>

using namespace cv;

// NN for Neural Network
class NNData {

    private:

    public:
        NNData();
        std::vector<Mat> LoadFromDirectory(std::string dir_path);

};
