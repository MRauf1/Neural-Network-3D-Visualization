#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>

using Eigen::MatrixXd;


// NN for Neural Network
class NN {

    private:

        int layers;
        std::vector<int> neurons;
        std::vector<MatrixXd> weights;
        std::vector<MatrixXd> biases;
        MatrixXd current_matrix;


    public:

        NN(std::vector<int> neurons);

        int GetLayers();

        std::vector<int> GetNeurons();

        std::vector<MatrixXd> GetWeights();

        std::vector<MatrixXd> GetBiases();

        void InitializeWeightsBiases(std::vector<int> neurons);

        double static Sigmoid(double num);

        MatrixXd ApplySigmoid(MatrixXd current_matrix);

};
