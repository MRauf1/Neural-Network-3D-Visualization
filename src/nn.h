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
        double threshold = 0.50;
        std::string kModelFolderPath = "data/model/";


    public:

        NN();

        /**
         * Training constructor for the Neural Network
         *
         * @param neurons       Vector containing the number of neurons
         *                      for each layer
         * @param learning_rate Learning rate for the training of the model
         */
        NN(std::vector<int> neurons, double learning_rate);

        /**
         * Getter for layers
         *
         * @return Number of layers
         */
        int GetLayers();

        /**
         * Getter for neurons
         *
         * @return Vector containing the number of neurons for each layer
         */
        std::vector<int> GetNeurons();

        /**
         * Getter for weights
         *
         * @return Vector containing weights of every layer
         */
        std::vector<MatrixXd> GetWeights();

        /**
         * Getter for biases
         *
         * @return Vector containing biases of every layer
         */
        std::vector<MatrixXd> GetBiases();

        /**
         * Getter for history
         *
         * @return Vector containing the pre-activation results of every layer
         */
        std::vector<MatrixXd> GetHistory();

        /**
         * Clears the vector containing the history needed for backpropagation
         */
        void ClearHistory();

        /**
         * Getter for activation_history
         *
         * @return Vector containing the post-activation results of every layer
         */
        std::vector<MatrixXd> GetActivationHistory();

        /**
         * Clears the vector containing the activation history needed
         * for backpropagation
         */
        void ClearActivationHistory();

        /**
         * Randomly initializes weights and biases of the model
         *
         * @param neurons Vector containing the number of neurons for each layer
         */
        void InitializeWeightsBiases(std::vector<int> neurons);

        /**
         * Single forward pass through the neural network
         *
         * @param  matrix Input matrix
         * @return        Output matrix
         */
        MatrixXd Feedforward(MatrixXd matrix);

        /**
         * Given a number, calculate its sigmoid
         *
         * @param  num The number that the function will be applied on
         * @return     Output of the function
         */
        double static Sigmoid(double num);

        /**
         * Given a number, calculate its sigmoid derivative
         *
         * @param  num The number that the function will be applied on
         * @return     Output of the function
         */
        double static SigmoidDerivative(double num);

        /**
         * Given a matrix, apply the sigmoid function element-wise
         *
         * @param  matrix Input matrix
         * @return        Output matrix
         */
        MatrixXd ApplySigmoid(MatrixXd matrix);

        /**
         * Given a matrix, apply the sigmoid derivative function element-wise
         *
         * @param  matrix Input matrix
         * @return        Output matrix
         */
        MatrixXd ApplySigmoidDerivative(MatrixXd matrix);

        /**
         * Given the labels and the predictions, calculate the Mean Squared
         * Error
         *
         * @param  labels      Vector of labels
         * @param  predictions Vector of predictions
         * @return             Mean Squared Error
         */
        double MSE(std::vector<int> labels, std::vector<MatrixXd> predictions);

        /**
         * Given the label and the prediction, calculate the Mean Squared
         * Error derivative
         *
         * @param  label      Label
         * @param  prediction Prediction in a matrix format
         * @return            Mean Squared Error derivative in matrix format
         */
        MatrixXd MSEDerivative(int label, MatrixXd prediction);

        /**
         * Apply the backpropagation algorithm
         *
         * @param label      Label of the image
         * @param prediction Prediction of the model in matrix format
         */
        void Backpropagation(int label, MatrixXd prediction);

        /**
         * Calculate the necessary updates for the weights and biases
         *
         * @param label      Label
         * @param prediction Prediction in matrix format
         * @return           Pair containing the updates for the weights
         *                   and the biases. First vector is for weights.
         *                   Second vector is for biases.
         */
        std::pair<std::vector<MatrixXd>, std::vector<MatrixXd>> CalculateErrors(int label, MatrixXd prediction);

        /**
         * Trains the model
         *
         * @param epochs Number of epochs to run for
         * @param images Vector containing the images
         * @param labels Vector containing the labels
         */
        void Train(int epochs, std::vector<MatrixXd> images, std::vector<int> labels);

        double Evaluate(std::vector<MatrixXd> images, std::vector<int> labels);

        /**
         * Save the weights and the biases of the model
         */
        void SaveModel();

        /**
         * Load the weights and the biases of the model
         */
        void LoadModel();

};
