#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <string>
#include <experimental/filesystem>

using namespace cv;
using Eigen::MatrixXd;


// NN for Neural Network
class NNData {

    private:

        std::string kDogInFileName = "dog";
        std::string kCatInFileName = "cat";
        int kLabelLength = 3;
        int kImageSize = 32;
        int kChannels = 3;
        int kNumPixels = kImageSize * kImageSize * kChannels;
        double kMaxPixelValue = 255.0;


    public:

        /**
         * Default constructor
         */
        NNData();

        /**
         * Given the directory path, load the data (images and labels) into
         * a pair.
         *
         * @param dir_path String of a directory path
         * @return         Pair containing the images and the labels
         */
        std::pair<std::vector<Mat>, std::vector<int>> LoadFromDirectory(std::string dir_path);

        /**
         * Given the filename, assign an appropriate label for the image
         *
         * @param  file_name String representing the local name of the file
         * @return           Integer label. 1 for dog, 0 for cat
         */
        int DogOrCat(std::string file_name);

        /**
         * Converts the vector of 3-channeled 2D images into a vector of
         * 1-channeled 1D images
         *
         * @param  images Vector containing the images
         * @return        Vector of reshaped images (images will be 1D
         *                (technically 2D, but there's one column, so
         *                it's like 1D))
         */
        std::vector<Mat> ConvertTo1D(std::vector<Mat> images);

        std::vector<MatrixXd> ConvertToEigen(std::vector<Mat>);

        std::vector<MatrixXd> Preprocess(std::vector<MatrixXd> images);

};
