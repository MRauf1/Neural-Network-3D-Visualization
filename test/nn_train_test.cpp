#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/nn.cpp"
#include "../src/nndata.cpp"

using namespace cv;
using Eigen::MatrixXd;

//int kImageSize = 100;
int kImageSize = 32;
int kChannels = 3;
int kNumTrain = 15000;
int kNumValidationTest = 5000;
int kNumPixels = kImageSize * kImageSize * kChannels;

NNData data;
//std::pair<std::vector<Mat>, std::vector<int>> train_data = data.LoadFromDirectory("../bin/data/train");
//std::pair<std::vector<Mat>, std::vector<int>> validation_data = data.LoadFromDirectory("../bin/data/validation");
std::pair<std::vector<Mat>, std::vector<int>> test_data = data.LoadFromDirectory("../bin/data/test");

//std::vector<int> neurons = {kNumPixels, 250, 10, 1};
std::vector<int> neurons = {kNumPixels, 750, 375, 1};
NN nn(neurons, 0.01);

TEST_CASE("Test train") {
    std::cout << "Starting the test case" << std::endl;
    std::vector<Mat> reshaped_data = data.ConvertTo1D(test_data.first);
    std::vector<MatrixXd> images = data.ConvertToEigen(reshaped_data);
    images = data.Preprocess(images);
    std::cout << "Starting training" << std::endl;
    nn.Train(5, images, test_data.second);
    std::vector<MatrixXd> weights = nn.GetWeights();
    std::vector<MatrixXd> biases = nn.GetBiases();

    for(int i = 0; i < weights.size(); i++) {
        std::cout << "Weight: " << i << std::endl;
        std::cout << weights.at(i) << std::endl;
        std::cout << "Bias: " << i << std::endl;
        std::cout << biases.at(i) << std::endl;
    }

}
