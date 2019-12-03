#include "nndata.h"
#include <experimental/filesystem>

using namespace cv;
namespace filesystem = std::experimental::filesystem;

NNData::NNData() {

}

std::vector<Mat> NNData::LoadFromDirectory(std::string dir_path) {

    std::vector<Mat> images;

    // Go through all images in the directory
    for (auto &file_path : filesystem::directory_iterator(dir_path)) {
        std::cout << file_path << std::endl;
        Mat image = imread(file_path.path().string());
        images.push_back(image);
    }

    return images;

}
