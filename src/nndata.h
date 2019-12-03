#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <experimental/filesystem>

using namespace cv;

// NN for Neural Network
class NNData {

    private:

        std::string kDogInFileName = "dog";
        std::string kCatInFileName = "cat";
        int kLabelLength = 3;


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

};
