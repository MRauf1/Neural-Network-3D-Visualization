#include "nndata.h"

using namespace cv;
using Eigen::MatrixXd;
//namespace filesystem = std::experimental::filesystem;

NNData::NNData() {

}

std::vector<std::string> NNData::GetImagePaths() {
    return image_paths;
}

std::pair<std::vector<Mat>, std::vector<int>> NNData::LoadFromDirectory(std::string dir_path) {

    std::vector<Mat> images;
    std::vector<int> image_labels;

    // Go through all images in the directory
    for(auto &file_path : std::experimental::filesystem::directory_iterator(dir_path)) {

        image_paths.push_back(file_path.path().string().substr(7));

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

    return -1;

}

std::vector<Mat> NNData::ConvertTo1D(std::vector<Mat> images) {

    std::vector<Mat> reshaped_images;

    for(Mat image : images) {
        reshaped_images.push_back(image.reshape(1, kNumPixels));
    }

    return reshaped_images;

}

// TODO: Make this more generalized
std::vector<MatrixXd> NNData::ConvertToEigen(std::vector<Mat> images) {

    std::vector<MatrixXd> images_matrix;

    for(Mat image : images) {
        MatrixXd image_matrix(kNumPixels, 1);
        cv2eigen(image, image_matrix);
        images_matrix.push_back(image_matrix);
    }

    return images_matrix;

}

std::vector<MatrixXd> NNData::Preprocess(std::vector<MatrixXd> images) {

    for(int index = 0; index < images.size(); index++) {
        images.at(index) /= kMaxPixelValue;
    }

    return images;

}
