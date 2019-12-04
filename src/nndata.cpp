#include "nndata.h"

using namespace cv;
using Eigen::MatrixXd;
namespace filesystem = std::experimental::filesystem;

NNData::NNData() {

}

std::pair<std::vector<Mat>, std::vector<int>> NNData::LoadFromDirectory(std::string dir_path) {

    std::vector<Mat> images;
    std::vector<int> image_labels;

    // Go through all images in the directory
    for(auto &file_path : filesystem::directory_iterator(dir_path)) {

        // Retrieve the image and resize it
        Mat image = imread(file_path.path().string());
        Mat resized_image;
        Size size(kImageSize, kImageSize);
        resize(image, resized_image, size);

        // Add to the respective vectors
        images.push_back(resized_image);
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

std::vector<Mat> NNData::ConvertTo1D(std::vector<Mat> images) {

    std::vector<Mat> reshaped_images;

    for(Mat image : images) {
        int new_num_rows = image.rows * image.cols * image.channels();
        reshaped_images.push_back(image.reshape(1, new_num_rows));
    }

    return reshaped_images;

}


void NNData::ConvertToEigen(std::pair<std::vector<Mat>, std::vector<int>>) {



}
