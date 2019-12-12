#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/nn.cpp"
#include "../src/nndata.cpp"
#include <fstream>

using namespace cv;
using Eigen::MatrixXd;

int kImageSize = 32;
int kChannels = 3;
int kNumTrain = 15000;
int kNumValidationTest = 5000;
int kNumPixels = kImageSize * kImageSize * kChannels;

NNData data;
//std::pair<std::vector<Mat>, std::vector<int>> train_data = data.LoadFromDirectory("../bin/data/train");
std::pair<std::vector<Mat>, std::vector<int>> validation_data = data.LoadFromDirectory("../bin/data/validation");
//std::pair<std::vector<Mat>, std::vector<int>> test_data = data.LoadFromDirectory("../bin/data/test");

std::vector<int> neurons = {kNumPixels, 750, 375, 1};
NN nn(neurons, 0.01);

TEST_CASE("Test model training") {
    std::cout << "Starting the test case" << std::endl;
    std::vector<Mat> reshaped_data = data.ConvertTo1D(validation_data.first);
    std::vector<MatrixXd> images = data.ConvertToEigen(reshaped_data);
    images = data.Preprocess(images);
    std::cout << "Starting training" << std::endl;
    nn.Train(5, images, validation_data.second);
    nn.SaveModel();
}
