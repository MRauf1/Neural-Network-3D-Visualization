#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

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

        MatrixXd Feedforward(MatrixXd matrix);

        double static Sigmoid(double num);

        double SigmoidDerivative(double num);

        MatrixXd ApplySigmoid(MatrixXd current_matrix);

        double MSE(std::vector<int> labels, std::vector<MatrixXd> predictions);

};
