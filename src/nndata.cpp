#include "nndata.h"
#include <experimental/filesystem>

using namespace cv;
namespace filesystem = std::experimental::filesystem;

NNData::NNData() {

}

std::pair<std::vector<Mat>, std::vector<int>> NNData::LoadFromDirectory(std::string dir_path) {

    std::vector<Mat> images;
    std::vector<int> image_labels;

    // Go through all images in the directory
    for (auto &file_path : filesystem::directory_iterator(dir_path)) {
        Mat image = imread(file_path.path().string());
        images.push_back(image);
        image_labels.push_back(DogOrCat(file_path.path().filename()));
    }

    // Create the pair of images and paths
    std::pair<std::vector<Mat>, std::vector<int>> data;
    data.first = images;
    data.second = image_labels;

    return data;

}

int NNData::DogOrCat(std::string file_name) {
    // Check if dog or cat based on the first 3 letters
    if(file_name.substr(0, kLabelLength).compare(kDogInFileName) == 0) {
        return 1;
    } else if(file_name.substr(0, kLabelLength).compare(kCatInFileName) == 0) {
        return 0;
    }

}
