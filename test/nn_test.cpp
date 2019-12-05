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
