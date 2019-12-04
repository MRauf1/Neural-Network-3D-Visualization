#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/nndata.cpp"

using namespace cv;

int kImageSize = 200;
int kChannels = 3;
int kNumTrain = 15000;
int kNumValidationTest = 5000;

NNData data;
std::pair<std::vector<Mat>, std::vector<int>> train_data = data.LoadFromDirectory("../bin/data/train");
std::pair<std::vector<Mat>, std::vector<int>> validation_data = data.LoadFromDirectory("../bin/data/validation");
std::pair<std::vector<Mat>, std::vector<int>> test_data = data.LoadFromDirectory("../bin/data/test");

// Testing proper loading of files
TEST_CASE("Check the number of training data") {
    REQUIRE(train_data.first.size() == kNumTrain);
    REQUIRE(train_data.second.size() == kNumTrain);
}

TEST_CASE("Check the number of validation data") {
    REQUIRE(validation_data.first.size() == kNumValidationTest);
    REQUIRE(validation_data.second.size() == kNumValidationTest);
}

TEST_CASE("Check the number of testing data") {
    REQUIRE(test_data.first.size() == kNumValidationTest);
    REQUIRE(test_data.second.size() == kNumValidationTest);
}

// Testing dataset balance
TEST_CASE("Check if the training data is balanced") {
    int num_dogs = 0;
    int num_cats = 0;
    std::vector<int> train_labels = train_data.second;
    for(int i = 0; i < train_labels.size(); i++) {
        if(train_labels[i] == 1) {
            num_dogs++;
        } else if(train_labels[i] == 0) {
            num_cats++;
        }
    }
    REQUIRE(num_dogs == (kNumTrain / 2));
    REQUIRE(num_cats == (kNumTrain / 2));
}

TEST_CASE("Check if the validation data is balanced") {
    int num_dogs = 0;
    int num_cats = 0;
    std::vector<int> validation_labels = validation_data.second;
    for(int i = 0; i < validation_labels.size(); i++) {
        if(validation_labels[i] == 1) {
            num_dogs++;
        } else if(validation_labels[i] == 0) {
            num_cats++;
        }
    }
    REQUIRE(num_dogs == (kNumValidationTest / 2));
    REQUIRE(num_cats == (kNumValidationTest / 2));
}

TEST_CASE("Check if the testing data is balanced") {
    int num_dogs = 0;
    int num_cats = 0;
    std::vector<int> test_labels = test_data.second;
    for(int i = 0; i < test_labels.size(); i++) {
        if(test_labels[i] == 1) {
            num_dogs++;
        } else if(test_labels[i] == 0) {
            num_cats++;
        }
    }
    REQUIRE(num_dogs == (kNumValidationTest / 2));
    REQUIRE(num_cats == (kNumValidationTest / 2));
}

// Testing DogOrCat method
TEST_CASE("Dog for DogOrCat") {
    REQUIRE(data.DogOrCat("dog123") == 1);
}

TEST_CASE("Cat for DogOrCat") {
    REQUIRE(data.DogOrCat("cat123") == 0);
}

// Testing ConvertTo1D
// Also trying to understand how OpenCV's reshape method works
TEST_CASE("Test image size after ConvertTo1D") {
    std::vector<Mat> reshaped_data = data.ConvertTo1D(test_data.first);
    Mat reshaped = reshaped_data.at(0);
    int reshaped_rows = reshaped_data.at(0).rows;
    REQUIRE(reshaped_rows == (kImageSize * kImageSize * kChannels));
}

TEST_CASE("Test col one after ConvertTo1D") {
    std::vector<Mat> reshaped_data = data.ConvertTo1D(test_data.first);
    Mat reshaped = reshaped_data.at(0);
    Mat image = test_data.first.at(0);
    int reshaped_col_one = reshaped.at<uchar>(0, 0);
    int col_one = (image.at<Vec3b>(0, 0)).val[0];
    REQUIRE(reshaped_col_one == col_one);
}

TEST_CASE("Test col two after ConvertTo1D") {
    std::vector<Mat> reshaped_data = data.ConvertTo1D(test_data.first);
    Mat reshaped = reshaped_data.at(0);
    Mat image = test_data.first.at(0);
    int reshaped_col_two = reshaped.at<uchar>(3, 0);
    int col_two = (image.at<Vec3b>(0, 1)).val[0];
    REQUIRE(reshaped_col_two == col_two);
}

TEST_CASE("Test channel two after ConvertTo1D") {
    std::vector<Mat> reshaped_data = data.ConvertTo1D(test_data.first);
    Mat reshaped = reshaped_data.at(0);
    Mat image = test_data.first.at(0);
    int reshaped_channel_two = reshaped.at<uchar>(1, 0);
    int channel_two = (image.at<Vec3b>(0, 0)).val[1];
    REQUIRE(reshaped_channel_two == channel_two);
}

TEST_CASE("Test consistency between images for ConvertTo1D") {
    std::vector<Mat> reshaped_data = data.ConvertTo1D(test_data.first);
    Mat reshaped_first = reshaped_data.at(0);
    Mat image_first = test_data.first.at(0);
    int reshaped_channel_two = reshaped_first.at<uchar>(1, 0);
    int channel_two = (image_first.at<Vec3b>(0, 0)).val[1];

    Mat reshaped_second = reshaped_data.at(1);
    Mat image_second = test_data.first.at(1);
    int reshaped_channel_two_second = reshaped_second.at<uchar>(1, 0);
    int channel_two_second = (image_second.at<Vec3b>(0, 0)).val[1];

    REQUIRE(reshaped_channel_two == channel_two);
    REQUIRE(reshaped_channel_two_second == channel_two_second);
}
