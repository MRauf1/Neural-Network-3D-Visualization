# Neural Network Visualization

A simple visualization of a basic deep neural network in 3D space with the ability to move around the space. The Neural Network is specifically designed to classify dogs and cats from the Kaggle dataset: https://www.kaggle.com/c/dogs-vs-cats/overview

### Dependencies
* OpenFrameworks
* OpenCV
* Eigen (Eigen's header files were used directly within the project)

### Quick Start
Clone the project. Navigate to the project folder. Then, run the following commands:
```console
$ make
$ cd bin
$ ./fantastic-finale-MRauf1
```

### Controls
| Key                       | Action                    |
| ------------------------- | ------------------------- |
| w                         | Move in positive y-axis   |
| s                         | Move in negative y-axis   |
| a                         | Move in negative x-axis   |
| d                         | Move in positive x-axis   |
| q                         | Move in negative z-axis   |
| e                         | Move in positive z-axis   |
| Mouse Click + Mouse Move  | Rotate                    |
