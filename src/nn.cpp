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

std::vector<MatrixXd> NN::GetHistory() {
    return this->history;
}

void NN::ClearHistory() {
    this->history.clear();
}

std::vector<MatrixXd> NN::GetActivationHistory() {
    return this->activation_history;
}

void NN::ClearActivationHistory() {
    this->activation_history.clear();
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

    // Store input for backpropagation
    this->activation_history.push_back(matrix);

    // Pass the input through every layer
    for(int layer = 0; layer < (this->layers - 1); layer++) {
        MatrixXd weight = this->weights.at(layer);
        MatrixXd bias = this->biases.at(layer);
        matrix = weight * matrix + bias;
        this->history.push_back(matrix); //***Clear in Backprop
        matrix = ApplySigmoid(matrix);
        this->activation_history.push_back(matrix);
    }

    this->current_matrix = matrix;
    return matrix;

}

double NN::Sigmoid(double num) {
    return (1.0 / (1.0 + exp(-num)));
}

double NN::SigmoidDerivative(double num) {
    return (Sigmoid(num) * (1 - Sigmoid(num)));
}

MatrixXd NN::ApplySigmoid(MatrixXd matrix) {
    return matrix.unaryExpr(&Sigmoid);
}

MatrixXd NN::ApplySigmoidDerivative(MatrixXd matrix) {
    return matrix.unaryExpr(&SigmoidDerivative);
}

double NN::MSE(std::vector<int> labels, std::vector<MatrixXd> predictions) {

    double error = 0;

    // Go through all predictions
    for(int index = 0; index < predictions.size(); index++) {
        error += pow((labels.at(index) - predictions.at(index)(0, 0)), 2);
    }

    // Get the average and return
    error /= (2.0 * predictions.size());
    return error;

}

MatrixXd NN::MSEDerivative(int label, MatrixXd prediction) {
    MatrixXd output(1, 1);
    output(0, 0) = prediction(0, 0) - label;
    return output;
}

//void NN::Backpropagation() {

//}

// NEEDS REFACTORING
std::pair<std::vector<MatrixXd>, std::vector<MatrixXd>> NN::CalculateErrors(int label, MatrixXd prediction) {

    std::vector<MatrixXd> weight_change(this->weights.size());
    std::vector<MatrixXd> bias_change(this->biases.size());

    // Layers are in backward order (last layer is first)
    std::vector<MatrixXd> layer_errors;
    int layer_iterator = this->history.size() - 1;

    // Calculate dC with respect to the output
    MatrixXd loss_gradient = MSEDerivative(label, prediction);

    // Calculate the derivative of the last activation layer
    MatrixXd output_layer_history = this->history.at(layer_iterator);
    MatrixXd sigmoid_derivative_vector = ApplySigmoidDerivative(output_layer_history);

    // Calculate the error of the last layer
    MatrixXd output_layer_error = loss_gradient.cwiseProduct(sigmoid_derivative_vector);
    layer_errors.push_back(output_layer_error);

    // Update the last weights and bias
    MatrixXd temp_weight = output_layer_error * this->activation_history.at(layer_iterator).transpose();
    MatrixXd temp_bias = output_layer_error;
    weight_change.at(layer_iterator) = temp_weight;
    bias_change.at(layer_iterator) = temp_bias;

    int layer_errors_iterator = 0;

    // Calculate the error for the rest of the layers (except for input layer)
    for(int layer = layer_iterator; layer > 0; layer--) {

        // Retrieve the necessary data
        MatrixXd transposed_weight = this->weights.at(layer).transpose();
        MatrixXd next_layer_error = layer_errors.at(layer_errors_iterator);
        layer_errors_iterator++;

        // Compute the error
        MatrixXd layer_error = transposed_weight * next_layer_error;

        // Calculate the derivative of the layer
        MatrixXd layer_history = this->history.at(layer - 1);
        MatrixXd sigmoid_derivative_vector = ApplySigmoidDerivative(layer_history);

        // Calculate the final error
        MatrixXd final_error = layer_error.cwiseProduct(sigmoid_derivative_vector);
        layer_errors.push_back(final_error);

        // Update the weights and bias
        MatrixXd temp_weight = final_error * this->activation_history.at(layer - 1).transpose();
        MatrixXd temp_bias = final_error;
        weight_change.at(layer - 1) = temp_weight;
        bias_change.at(layer - 1) = temp_bias;

    }

    std::pair<std::vector<MatrixXd>, std::vector<MatrixXd>> changes = {weight_change, bias_change};

    return changes;

}
