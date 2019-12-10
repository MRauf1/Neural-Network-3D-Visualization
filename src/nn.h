#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <random>

using Eigen::MatrixXd;


// NN for Neural Network
class NN {

    private:

        int layers;
        std::vector<int> neurons;
        std::vector<MatrixXd> weights;
        std::vector<MatrixXd> biases;
        MatrixXd current_matrix;
        std::vector<MatrixXd> history;
        std::vector<MatrixXd> activation_history;
        double learning_rate;


    public:

        NN(std::vector<int> neurons, double learning_rate);

        int GetLayers();

        std::vector<int> GetNeurons();

        std::vector<MatrixXd> GetWeights();

        std::vector<MatrixXd> GetBiases();

        std::vector<MatrixXd> GetHistory();

        void ClearHistory();

        std::vector<MatrixXd> GetActivationHistory();

        void ClearActivationHistory();

        void InitializeWeightsBiases(std::vector<int> neurons);

        MatrixXd Feedforward(MatrixXd matrix);

        double static Sigmoid(double num);

        double static SigmoidDerivative(double num);

        MatrixXd ApplySigmoid(MatrixXd matrix);

        MatrixXd ApplySigmoidDerivative(MatrixXd matrix);

        double MSE(std::vector<int> labels, std::vector<MatrixXd> predictions);

        MatrixXd MSEDerivative(int label, MatrixXd prediction);

        void Backpropagation(int label, MatrixXd prediction);

        std::pair<std::vector<MatrixXd>, std::vector<MatrixXd>> CalculateErrors(int label, MatrixXd prediction);

        void Train(int epochs, std::vector<MatrixXd> images, std::vector<int> labels);

};
