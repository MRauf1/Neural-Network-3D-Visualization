#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/nn.cpp"

using Eigen::MatrixXd;

std::vector<int> neurons = {3, 2, 3};
NN simple_test_nn(neurons, 0.1);

std::vector<int> neurons_two = {3, 2, 1};
NN nn_two(neurons_two, 0.1);

// Testing InitializeWeightsBiases
TEST_CASE("Check the weights vector size") {
    std::vector<MatrixXd> weights = simple_test_nn.GetWeights();
    REQUIRE(weights.size() == 2);
}

TEST_CASE("Check the first weights' matrix size") {
    std::vector<MatrixXd> weights = simple_test_nn.GetWeights();
    MatrixXd sample_weight = weights.at(0);
    REQUIRE(sample_weight.rows() == 2);
    REQUIRE(sample_weight.cols() == 3);
}

TEST_CASE("Check the second weights' matrix size") {
    std::vector<MatrixXd> weights = simple_test_nn.GetWeights();
    MatrixXd sample_weight = weights.at(1);
    REQUIRE(sample_weight.rows() == 3);
    REQUIRE(sample_weight.cols() == 2);
}

TEST_CASE("Check the weights randomness") {
    std::vector<MatrixXd> weights = simple_test_nn.GetWeights();
    MatrixXd sample_weight = weights.at(0);
    REQUIRE(sample_weight(0, 0) != sample_weight(0, 1));
}

TEST_CASE("Check the biases size") {
    std::vector<MatrixXd> biases = simple_test_nn.GetBiases();
    MatrixXd sample_bias = biases.at(0);
    REQUIRE(sample_bias.rows() == 2);
    REQUIRE(sample_bias.cols() == 1);
}

// Check the constructor
TEST_CASE("Check the constructor") {
    int layers = simple_test_nn.GetLayers();
    std::vector<int> neurons = simple_test_nn.GetNeurons();
    REQUIRE(layers == 3);
    REQUIRE(neurons.size() == 3);
    REQUIRE(neurons.at(0) == 3);
    REQUIRE(neurons.at(1) == 2);
    REQUIRE(neurons.at(2) == 3);
}

// Check Feedforward method
// WARNING: Correctness of calculations of Feedforward
// is not tested. Potentially implement it.
TEST_CASE("Check matrix size after Feedforward") {
    MatrixXd sample_matrix = MatrixXd::Random(3, 1);
    MatrixXd result = simple_test_nn.Feedforward(sample_matrix);
    REQUIRE(result.rows() == 3);
    REQUIRE(result.cols() == 1);
    simple_test_nn.ClearHistory();
    simple_test_nn.ClearActivationHistory();
}

// Check history
TEST_CASE("Check history size after Feedforward") {
    MatrixXd sample_matrix = MatrixXd::Random(3, 1);
    MatrixXd result = simple_test_nn.Feedforward(sample_matrix);
    std::vector<MatrixXd> history = simple_test_nn.GetHistory();
    REQUIRE(history.size() == 2);
    simple_test_nn.ClearHistory();
    simple_test_nn.ClearActivationHistory();
}

TEST_CASE("Check hidden layer history after Feedforward") {
    MatrixXd sample_matrix = MatrixXd::Random(3, 1);
    MatrixXd result = simple_test_nn.Feedforward(sample_matrix);
    std::vector<MatrixXd> history = simple_test_nn.GetHistory();
    MatrixXd hidden_layer_history = history.at(0);
    REQUIRE(hidden_layer_history.rows() == 2);
    REQUIRE(hidden_layer_history.cols() == 1);
    simple_test_nn.ClearHistory();
    simple_test_nn.ClearActivationHistory();
}

TEST_CASE("Check output layer history after Feedforward") {
    MatrixXd sample_matrix = MatrixXd::Random(3, 1);
    MatrixXd result = simple_test_nn.Feedforward(sample_matrix);
    std::vector<MatrixXd> history = simple_test_nn.GetHistory();
    MatrixXd output_layer_history = history.at(1);
    REQUIRE(output_layer_history.rows() == 3);
    REQUIRE(output_layer_history.cols() == 1);
    simple_test_nn.ClearHistory();
    simple_test_nn.ClearActivationHistory();
}

// Check activation_history
TEST_CASE("Check activation history size after Feedforward") {
    MatrixXd sample_matrix = MatrixXd::Random(3, 1);
    MatrixXd result = simple_test_nn.Feedforward(sample_matrix);
    std::vector<MatrixXd> activation_history = simple_test_nn.GetActivationHistory();
    REQUIRE(activation_history.size() == 3);
    simple_test_nn.ClearHistory();
    simple_test_nn.ClearActivationHistory();
}

TEST_CASE("Check activation history of input layer after Feedforward") {
    MatrixXd sample_matrix = MatrixXd::Random(3, 1);
    MatrixXd result = simple_test_nn.Feedforward(sample_matrix);
    std::vector<MatrixXd> activation_history = simple_test_nn.GetActivationHistory();
    MatrixXd activation = activation_history.at(0);
    REQUIRE(activation.rows() == 3);
    REQUIRE(activation.cols() == 1);
    simple_test_nn.ClearHistory();
    simple_test_nn.ClearActivationHistory();
}

TEST_CASE("Check activation history of hidden layer after Feedforward") {
    MatrixXd sample_matrix = MatrixXd::Random(3, 1);
    MatrixXd result = simple_test_nn.Feedforward(sample_matrix);
    std::vector<MatrixXd> activation_history = simple_test_nn.GetActivationHistory();
    MatrixXd activation = activation_history.at(1);
    REQUIRE(activation.rows() == 2);
    REQUIRE(activation.cols() == 1);
    simple_test_nn.ClearHistory();
    simple_test_nn.ClearActivationHistory();
}

TEST_CASE("Check activation history of output layer after Feedforward") {
    MatrixXd sample_matrix = MatrixXd::Random(3, 1);
    MatrixXd result = simple_test_nn.Feedforward(sample_matrix);
    std::vector<MatrixXd> activation_history = simple_test_nn.GetActivationHistory();
    MatrixXd activation = activation_history.at(2);
    REQUIRE(activation.rows() == 3);
    REQUIRE(activation.cols() == 1);
    simple_test_nn.ClearHistory();
    simple_test_nn.ClearActivationHistory();
}

// Check Sigmoid method
TEST_CASE("Check the Sigmoid") {
    double num = 3.5;
    double result = simple_test_nn.Sigmoid(num);
    REQUIRE((result > 0.97 && result < 0.98) == true);
}

// Check SigmoidDerivative method
TEST_CASE("Check the SigmoidDerivative") {
    double num = 3.5;
    double result = simple_test_nn.SigmoidDerivative(num);
    REQUIRE((result > 0.028 && result < 0.029) == true);
}

// Check ApplySigmoid method
// Checked this manually too
TEST_CASE("Check the ApplySigmoid") {
    MatrixXd matrix = MatrixXd::Random(2, 2);
    MatrixXd result = simple_test_nn.ApplySigmoid(matrix);
    REQUIRE((result(1, 1) >= 0) == true);
}

// Check ApplySigmoidDerivative method
// Checked this manually too
TEST_CASE("Check the ApplySigmoidDerivative") {
    MatrixXd matrix = MatrixXd::Random(4, 1);
    MatrixXd result = simple_test_nn.ApplySigmoidDerivative(matrix);
    REQUIRE((result(0, 0) >= 0) == true);
}

// Check MSE method
TEST_CASE("Check the MSE method") {
    std::vector<int> labels = {0, 1};
    MatrixXd prediction_one(1, 1);
    prediction_one << 0.3;
    MatrixXd prediction_two(1, 1);
    prediction_two << 0.9;
    std::vector<MatrixXd> predictions = {prediction_one, prediction_two};
    double error = simple_test_nn.MSE(labels, predictions);
    REQUIRE((error > 0.024 && error < 0.026) == true);
}

// Check MSEDerivative method
TEST_CASE("Check the MSEDerivative method") {
    int label = 1;
    MatrixXd prediction(1, 1);
    prediction << 0.3;
    MatrixXd error_derivative = simple_test_nn.MSEDerivative(label, prediction);
    double value = error_derivative(0, 0);
    REQUIRE(error_derivative.rows() == 1);
    REQUIRE(error_derivative.cols() == 1);
    REQUIRE((value < -0.69 && value > -0.71) == true);
}

// Check CalculateErrors method
// NOT FINISHED
TEST_CASE("Check the error of output layer") {
    MatrixXd sample_matrix = MatrixXd::Random(3, 1);
    MatrixXd result = nn_two.Feedforward(sample_matrix);
    int label = 1;
    nn_two.CalculateErrors(label, result);
    nn_two.ClearHistory();
    nn_two.ClearActivationHistory();
}

// Check Backpropagation method
// NOT FINISHED
TEST_CASE("Check the error of output layer 2") {
    MatrixXd sample_matrix = MatrixXd::Random(3, 1);
    MatrixXd result = nn_two.Feedforward(sample_matrix);
    int label = 1;
    nn_two.Backpropagation(label, result);
}
