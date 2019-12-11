#include "nn_visualization.h"

NNVisualization::NNVisualization() {

}

void NNVisualization::Initialize(std::vector<std::string> image_paths,
            std::vector<MatrixXd> images, std::vector<int> labels, NN nn) {
    this->image_paths = image_paths;
    this->images = images;
    this->labels = labels;
    this->nn = nn;
}

void NNVisualization::PassFeedForward() {
    MatrixXd output = nn.Feedforward(images.at(index));
    this->history =  nn.GetHistory();
    this->activation_history = nn.GetActivationHistory();
}

void NNVisualization::DrawImage() {
    image.load(image_paths.at(index));
    image.draw(0, 0, kLayerDistance);
    index++;
}

void NNVisualization::DrawAllLayers() {
    std::cout << "Starting to draw the layers" << std::endl;
    std::vector<int> neurons = nn.GetNeurons();

    for(int layer = 0; layer < neurons.size(); layer++) {

        DrawLayer(layer, neurons.at(layer));

    }
    std::cout << "Finished drawing the layers" << std::endl;
}

void NNVisualization::DrawLayer(int layer, int num_neurons) {
    int cols_per_row = GetSmallestSumFactor(num_neurons);
    for(int row = 0; row < (num_neurons / cols_per_row); row++) {
        for(int col = 0; col < cols_per_row; col++) {
            ofDrawIcoSphere(col * kNeuronDistance, row * kNeuronDistance, layer * -kLayerDistance, kNeuronRadius);
        }
    }

}

int NNVisualization::GetSmallestSumFactor(int num) {

    if(num == 1) {
        return num;
    }

    int smallest_sum = INT_MAX;

    // Start from half and go down
    for(int factor = (num / 2); factor > 1; factor--) {
        if(num % factor == 0 && (factor + (num / factor)) < smallest_sum) {
            smallest_sum = factor;
        }
    }

    return smallest_sum;

}
