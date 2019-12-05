#include "nn.h"

using Eigen::MatrixXd;

NN::NN(std::vector<int> neurons) {
    this->layers = neurons.size();
    this->neurons = neurons;
    InitializeWeightsBiases(neurons);
}

int NN::GetLayers() {
    return this->layers;
}

std::vector<int> NN::GetNeurons() {
    return this->neurons;
}

std::vector<MatrixXd> NN::GetWeights() {
    return this->weights;
}

std::vector<MatrixXd> NN::GetBiases() {
    return this->biases;
}

void NN::InitializeWeightsBiases(std::vector<int> neurons) {

    // Initialize for every layer
    for(int index = 0; index < (neurons.size() - 1); index++) {
        // Initialize weights between the consecutive layer pairs
        MatrixXd layer_weights = MatrixXd::Random(neurons.at(index + 1), neurons.at(index));
        this->weights.push_back(layer_weights);
        // Initialize biases for every layer except for the input layer
        MatrixXd layer_biases = MatrixXd::Random(neurons.at(index + 1), 1);
        this->biases.push_back(layer_biases);
    }

}

MatrixXd NN::Feedforward(MatrixXd matrix) {

    for(int layer = 0; layer < (this->layers - 1); layer++) {
        MatrixXd weight = this->weights.at(layer);
        MatrixXd bias = this->biases.at(layer);
        matrix = weight * matrix + bias;
        matrix = ApplySigmoid(matrix);
    }

    this->current_matrix = matrix;
    return matrix;

}

double NN::Sigmoid(double num) {
    return (1.0 / (1.0 + exp(-num)));
}

MatrixXd NN::ApplySigmoid(MatrixXd matrix) {
    return matrix.unaryExpr(&Sigmoid);
}
