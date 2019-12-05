#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "../src/nn.cpp"

using Eigen::MatrixXd;

std::vector<int> neurons = {3, 2, 3};
NN simple_test_nn(neurons);

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
}

// Check Sigmoid method
TEST_CASE("Check the Sigmoid") {
    double num = 3.5;
    double result = simple_test_nn.Sigmoid(num);
    REQUIRE((result > 0.97 && result < 0.98) == true);
}

// Check ApplySigmoid method
// Checked this manually too
TEST_CASE("Check the ApplySigmoid") {
    MatrixXd matrix = MatrixXd::Random(2, 2);
    MatrixXd result = simple_test_nn.ApplySigmoid(matrix);
    REQUIRE((result(1, 1) >= 0) == true);
}
