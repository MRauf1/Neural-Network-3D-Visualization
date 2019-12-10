**11/23/19**\
-Created the project\
-Installed OpenCV for the project\
-Downloaded the dataset of cats and dogs\
-Separated the dataset into training, validation, and testing images

**11/24/19**\
-Started working on the class NNData used to store and process the data for the neural network\
-Wrote the code for loading in images from folders

**12/3/19**\
-Tested the code for loading in images from folders\
-Modified the loading function to fetch both the images and their labels\
-Tested the balance of the dataset\
-Wrote a helper method for assigning labels based on the file names\
-Tested the helper method\
-Wrote the method for converting images to 1D vectors (necessary for the simple neural network architecture)\
-Tested the converter\
-Installed Eigen for the project\
-Set the resize resolution to 200x200

**12/4/19**\
-Wrote the method for converting the OpenCV images into Eigen matrices\
-Tested the converter\
-Wrote the method for preprocessing the data\
-Tested the preprocessing method\
-Changed the resize resolution to 100x100

**12/5/19**\
-Created the file for the Neural Network itself\
-Began working on the Neural Network\
-Implemented the constructor\
-Implemented the getters\
-Implemented the method for initializing weights and biases of the Neural Network\
-Implemented the element Sigmoid function\
-Implemented the matrix Sigmoid function\
-Tested both of the Sigmoid functions\
-Programmed the Feedforward portion of the Network\
-Partially tested the Feedforward method\
-Implemented the Mean Squared Error function\
-Tested the MSE function

**12/6/19**\
-Coded the function for calculating the derivative of Sigmoid\
-Tested the Sigmoid derivative\
-Added code for storing the history for backpropagation\
-Tested the new code\
-Added the helper method for clearing the history

**12/7/19**
-Coded the function for computing the derivative of the loss function\
-Tested the loss derivative\
-Coded the function for applying the sigmoid derivative over the whole matrix\
-Tested the sigmoid derivative function\

**12/8/19**
-Added a variable for keeping track of activation history\
-Modified Feedforward to update the activation history\
-Wrapped up with the CalculateErrors method, which is used to find the desired changes to the weights and biases
