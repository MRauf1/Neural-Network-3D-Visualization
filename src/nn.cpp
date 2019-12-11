#include "nn.h"

using Eigen::MatrixXd;

NN::NN() {

}

NN::NN(std::vector<int> neurons, double learning_rate) {
    this->layers = neurons.size();
    this->neurons = neurons;
    InitializeWeightsBiases(neurons);
    this->learning_rate = learning_rate;
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

        // Retrieve the necessary weights and biases
        MatrixXd weight = this->weights.at(layer);
        MatrixXd bias = this->biases.at(layer);

        // Calculate the output and store the necessary calculations
        matrix = weight * matrix + bias;
        this->history.push_back(matrix);
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

void NN::Backpropagation(int label, MatrixXd prediction) {

    // Retrieve the changes
    std::pair<std::vector<MatrixXd>, std::vector<MatrixXd>> changes = CalculateErrors(label, prediction);
    std::vector<MatrixXd> weight_change = changes.first;
    std::vector<MatrixXd> bias_change = changes.second;

    // Go through each layer and update every weight and bias
    for(int layer = 0; layer < this->weights.size(); layer++) {

        this->weights.at(layer) -= (learning_rate * weight_change.at(layer));
        this->biases.at(layer) -= (learning_rate * bias_change.at(layer));

    }

    ClearHistory();
    ClearActivationHistory();

}

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

void NN::Train(int epochs, std::vector<MatrixXd> images, std::vector<int> labels) {

    // Shuffle the images and the labels
    int seed = 1;
    shuffle(images.begin(), images.end(), std::default_random_engine(seed));
    shuffle(labels.begin(), labels.end(), std::default_random_engine(seed));

    // Train the model for the defined number of epochs
    for(int epoch = 0; epoch < epochs; epoch++) {

        std::cout << "Epoch: " << epoch << std::endl;
        double error_total = 0;
        double accuracy = 0;

        // For each epoch, train on every available image
        for(int image_num = 0; image_num < images.size(); image_num++) {

            MatrixXd prediction = Feedforward(images.at(image_num));
            std::vector<int> label = {labels.at(image_num)};
            std::vector<MatrixXd> prediction_two = {prediction};

            error_total += MSE(label, prediction_two);
            double error_average = (error_total / (image_num + 1));
            std::cout << "Image: " << image_num << " Error: " << error_average << std::endl;

            if((prediction(0, 0) >= threshold && label.at(0) == 1) ||
                (prediction(0, 0) < threshold && label.at(0) == 0)) {
                accuracy += 1;
            }

            double accuracy_average = (accuracy / (image_num + 1));
            std::cout << "Accuracy: " << accuracy_average << std::endl;

            Backpropagation(labels.at(image_num), prediction);

        }

        error_total = 0;
        accuracy = 0;

    }

}

double NN::Evaluate(std::vector<MatrixXd> images, std::vector<int> labels) {

    double accuracy = 0;

    // Go through every image
    for(int image_num = 0; image_num < images.size(); image_num++) {

        std::cout << "Image Number: " << image_num << std::endl;

        // Get the prediction and the label
        MatrixXd prediction = Feedforward(images.at(image_num));
        std::vector<int> label = {labels.at(image_num)};

        // Check if the image was predicted correctly
        if((prediction(0, 0) >= threshold && label.at(0) == 1) ||
            (prediction(0, 0) < threshold && label.at(0) == 0)) {
            accuracy += 1;
        }

    }

    return (accuracy / (images.size()));

}

void NN::SaveModel() {

    // Go through all layers
    for(int num = 0; num < this->weights.size(); num++) {

        // Save the weights
        std::string weight_file_name = kModelFolderPath + "weight_" + std::to_string(num) + ".txt";
        std::ofstream weight_file(weight_file_name);
        MatrixXd weight = this->weights.at(num);

        if(weight_file.is_open()) {
            weight_file << weight;
        }

        // Save the biases
        std::string bias_file_name = kModelFolderPath + "bias_" + std::to_string(num) + ".txt";
        std::ofstream bias_file(bias_file_name);
        MatrixXd bias = this->biases.at(num);

        if(bias_file.is_open()) {
            bias_file << bias;
        }

    }

}

// NEEDS REFACTORING
void NN::LoadModel() {

    std::vector<MatrixXd> temp_weights(this->layers - 1);
    std::vector<MatrixXd> temp_biases(this->layers - 1);

    // Go through all images in the directory
    for(auto &file_path : std::experimental::filesystem::directory_iterator(kModelFolderPath)) {

        std::cout << file_path.path().string() << std::endl;

        // Get the file and initialize the necessary variables
        std::string file_name = file_path.path().string();
        std::ifstream file(file_name);
        std::string file_content;
        std::string line;
        std::vector<std::string> result;
        std::regex regex("\\s+");

        // Get the needed values for later storing the matrices in correct order
        bool is_weight = (file_name.substr(11, 6).compare("weight") == 0);
        std::regex num_regex("\\d+");
        std::smatch num_regex_result;
        std::regex_search(file_name, num_regex_result, num_regex);
        int index = std::stoi(num_regex_result[0]);

        // Initialize the matrix for storing the data
        MatrixXd matrix;

        if(is_weight) {
            matrix.resize(this->weights.at(index).rows(), this->weights.at(index).cols());
        } else {
            matrix.resize(this->biases.at(index).rows(), this->biases.at(index).cols());
        }

        int row = 0;

        if(file.is_open()) {

            while(getline(file,line)) {

                // Remove leading, trailing, extra spaces
                line = std::regex_replace(line, std::regex("^ +| +$|( ) +"), "$1");
                std::sregex_token_iterator iterator(line.begin(), line.end(), regex, -1);
                std::sregex_token_iterator end;

                int col = 0;

                // Add the columns to the matrix
                while(iterator != end) {
                    matrix(row, col) = std::stod((*iterator));
                    ++iterator;
                    col++;
                }

                row++;

            }

            file.close();

        }

        if(is_weight) {
            temp_weights.at(index) = matrix;
        } else {
            temp_biases.at(index) = matrix;
        }

    }

    this->weights = temp_weights;
    this->biases = temp_biases;

}
