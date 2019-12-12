#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <random>
#include <fstream>
#include <experimental/filesystem>
#include <regex>
#include "ofMain.h"
#include "nn.h"

using Eigen::MatrixXd;


// NN for Neural Network
class NNVisualization {

    private:

        std::vector<MatrixXd> images;
        std::vector<int> labels;
        std::vector<std::string> image_paths;
        std::vector<MatrixXd> history;
        std::vector<MatrixXd> activation_history;
        MatrixXd previous_layer_x_pos;
        MatrixXd previous_layer_y_pos;
        NN nn;
        ofImage image;
        int index = 0;
        int kImageSize = 32;
        int kLayerDistance = 100;
        int kNeuronDistance = 20;
        int kNeuronRadius = 5;
        int kDisplayImageSize = 500;
        int kMaxPixelValue = 255;
        double kThreshold = 0.50;


    public:

        /**
         * Default constructor
         */
        NNVisualization();

        /**
         * Getter for index
         *
         * @return Image number that is currently processed
         */
        int GetIndex();

        /**
         * Setter for index
         *
         * @param index Image number that is currently processed
         */
        void SetIndex(int index);

        /**
         * Initializes the visualization object
         *
         * @param image_paths Vector with string paths of the images
         * @param images      Vector with the images
         * @param labels      Vector with the int labels
         * @param nn          The Neural Network used for predictions
         */
        void Initialize(std::vector<std::string> image_paths, std::vector<MatrixXd> images, std::vector<int> labels, NN nn);

        /**
         * Passes the image determined by index through the network to
         * get a prediction
         */
        void PassFeedforward();

        /**
         * Draws the predicted label
         */
        void DrawLabel();

        /**
         * Draws the input image
         */
        void DrawImage();

        /**
         * Draws all the layers of the network
         */
        void DrawAllLayers();

        /**
         * Draws a single layer of the network
         *
         * @param layer       Int representing the layer number
         * @param num_neurons Number of neurons in the layer
         */
        void DrawLayer(int layer, int num_neurons);

        /**
         * Given a number, find the factor pair which produces the
         * smallest sum
         *
         * @param  num Number for analysis
         * @return     One of the factors
         */
        int GetSmallestSumFactor(int num);

};
