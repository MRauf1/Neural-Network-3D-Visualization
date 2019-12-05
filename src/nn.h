#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

using Eigen::MatrixXd;


// NN for Neural Network
class NN {

    private:

        int layers;
        std::vector<int> neurons;
        std::vector<MatrixXd> weights;
        std::vector<MatrixXd> biases;


    public:

        NN(std::vector<int> neurons);

        int GetLayers();

        std::vector<int> GetNeurons();

        std::vector<MatrixXd> GetWeights();

        std::vector<MatrixXd> GetBiases();

        void InitializeWeightsBiases(std::vector<int> neurons);

};
