#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/nndata.cpp"

using namespace cv;

NNData data;
std::pair<std::vector<Mat>, std::vector<int>> train_data = data.LoadFromDirectory("../bin/data/train");
std::pair<std::vector<Mat>, std::vector<int>> validation_data = data.LoadFromDirectory("../bin/data/validation");
std::pair<std::vector<Mat>, std::vector<int>> test_data = data.LoadFromDirectory("../bin/data/test");

// Testing proper loading of files
TEST_CASE("Check the number of training data") {
    REQUIRE(train_data.first.size() == 15000);
    REQUIRE(train_data.second.size() == 15000);
}

TEST_CASE("Check the number of validation data") {
    REQUIRE(validation_data.first.size() == 5000);
    REQUIRE(validation_data.second.size() == 5000);
}

TEST_CASE("Check the number of testing data") {
    REQUIRE(test_data.first.size() == 5000);
    REQUIRE(test_data.second.size() == 5000);
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
    REQUIRE(num_dogs == 7500);
    REQUIRE(num_cats == 7500);
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
    REQUIRE(num_dogs == 2500);
    REQUIRE(num_cats == 2500);
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
    REQUIRE(num_dogs == 2500);
    REQUIRE(num_cats == 2500);
}

// Testing DogOrCat method
TEST_CASE("Dog for DogOrCat") {
    REQUIRE(data.DogOrCat("dog123") == 1);
}

TEST_CASE("Cat for DogOrCat") {
    REQUIRE(data.DogOrCat("cat123") == 0);
}
