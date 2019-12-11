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

void NNVisualization::DrawImage() {
    image.load(image_paths.at(index));
    image.draw(0, 0, kImageSize, kImageSize);
    index++;
}

//void NNVisualization::DrawLayer() {



//}
