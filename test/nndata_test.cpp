#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/nndata.cpp"

using namespace cv;

NNData data;
std::vector<Mat> train_images = data.LoadFromDirectory("../bin/data/train");
std::vector<Mat> validation_images = data.LoadFromDirectory("../bin/data/validation");
std::vector<Mat> test_images = data.LoadFromDirectory("../bin/data/test");


TEST_CASE("Check the number of training images") {
    REQUIRE(train_images.size() == 15000);
}

TEST_CASE("Check the number of validation images") {
    REQUIRE(validation_images.size() == 5000);
}

TEST_CASE("Check the number of testing images") {
    REQUIRE(test_images.size() == 5000);
}

TEST_CASE("Check if the training images are balanced") {
    int num_dogs = 0;
    int num_cats = 0;
    for(int i = 0; i < train_images.size(); i++) {
        if() {

        } else if() {

        }
    }


    REQUIRE(test_images.size() == 5000);
}
