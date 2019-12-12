#include "nn_visualization.h"

NNVisualization::NNVisualization() {

}

int NNVisualization::GetIndex() {
    return this->index;
}

void NNVisualization::SetIndex(int index) {
    this->index = index;
}


void NNVisualization::Initialize(std::vector<std::string> image_paths,
            std::vector<MatrixXd> images, std::vector<int> labels, NN nn) {

    // Initialize the object
    this->image_paths = image_paths;
    this->images = images;
    this->labels = labels;
    this->nn = nn;

}

void NNVisualization::PassFeedforward() {

    // Predict on the image
    MatrixXd output = nn.Feedforward(images.at(index));
    this->history =  nn.GetHistory();
    this->activation_history = nn.GetActivationHistory();

}

void NNVisualization::DrawLabel() {

    // Retrieve the output
    MatrixXd output = this->activation_history.at(this->activation_history.size() - 1);
    std::string label;

    // Get the label
    if(output(0, 0) >= kThreshold) {
        label = "Dog";
    } else {
        label = "Cat";
    }

    // Draw the label
    ofDrawBitmapString(label, 0, kDisplayImageSize, kLayerDistance);

}

void NNVisualization::DrawImage() {
    ofSetColor(kMaxPixelValue, kMaxPixelValue, kMaxPixelValue);
    image.load(image_paths.at(index));
    image.draw(0, 0, kLayerDistance, kDisplayImageSize, kDisplayImageSize);
}

void NNVisualization::DrawAllLayers() {

    std::vector<int> neurons = nn.GetNeurons();

    // Draw each layer of the network
    for(int layer = 0; layer < neurons.size(); layer++) {
        DrawLayer(layer, neurons.at(layer));
    }

    // Clear the history and increment the index to get the next image
    nn.ClearHistory();
    nn.ClearActivationHistory();
    index++;

}

void NNVisualization::DrawLayer(int layer, int num_neurons) {

    // Get the values necessary to draw the layer
    MatrixXd activation = this->activation_history.at(layer);
    int cols_per_row = GetSmallestSumFactor(num_neurons);
    int total_rows = num_neurons / cols_per_row;

    // Offsets to make sure each layer is centered at the image's center
    int x_offset = -(cols_per_row * kNeuronDistance / 2) + (kDisplayImageSize / 2);
    int y_offset =  -(total_rows * kNeuronDistance / 2) + (kDisplayImageSize / 2);

    // Draw all the neurons of the layer
    for(int row = 0; row < total_rows; row++) {

        for(int col = 0; col < cols_per_row; col++) {

            // Retrieve the activation value for the neuron and draw a
            // sphere representing the neuron with the color of the sphere
            // indicating the magnitude of the activation value
            double activation_value = activation(row * total_rows + col, 0);
            int pixel_value = int(activation_value * kMaxPixelValue);
            ofSetColor(pixel_value, pixel_value, pixel_value);
            int x_pos = col * kNeuronDistance + x_offset;
            int y_pos = row * kNeuronDistance + y_offset;
            ofDrawIcoSphere(x_pos, y_pos, layer * -kLayerDistance, kNeuronRadius);

            // Draw the lines which represent the weights/biases
            // WARNING: Since the trained model has 2-3 million weights and biases
            // combined, this code will try to draw all of them.
            // Due to the sheer amount of computation needed for this,
            // the program is likely to crash if this code is uncommented.
            // Run this at your own risk.

            /*
            if(layer != 0) {
                for(int prev_row = 0; prev_row < this->previous_layer_x_pos.rows(); prev_row++) {
                    for(int prev_col = 0; prev_col < this->previous_layer_x_pos.cols(); prev_row++) {
                        ofSetColor(kMaxPixelValue, kMaxPixelValue, kMaxPixelValue);
                        ofDrawLine(this->previous_layer_x_pos(prev_row, prev_col), this->previous_layer_y_pos(prev_row, prev_col), (layer - 1) * -kLayerDistance, x_pos, y_pos, layer * -kLayerDistance);
                    }
                }
            }

            // Store x coordinates
            MatrixXd temp_previous_layer_x_pos(total_rows, cols_per_row);
            temp_previous_layer_x_pos(row, col) = x_pos;
            this->previous_layer_x_pos = temp_previous_layer_x_pos;

            // Store y coordinates
            MatrixXd temp_previous_layer_y_pos(total_rows, cols_per_row);
            temp_previous_layer_y_pos(row, col) = y_pos;
            this->previous_layer_y_pos = temp_previous_layer_y_pos;
            */

        }

    }

}

int NNVisualization::GetSmallestSumFactor(int num) {

    // Return if the number is small
    if(num <= 3) {
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
