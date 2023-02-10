
## Terms Used

- Feature: The input(s) to our model
- Examples: An input/output pair used for training
- Labels: The output of the model
- Layer: A collection of nodes connected together within a neural network.
- Model: The representation of your neural network
- Dense and Fully Connected (FC): Each node in one layer is connected to each node in the previous layer.
- Weights and biases: The internal variables of model
- Loss: The discrepancy between the desired output and the actual output
- MSE: Mean squared error, a type of loss function that counts a small number of large discrepancies as worse than a large number of small ones.
- Gradient Descent: An algorithm that changes the internal variables a bit at a time to gradually reduce the loss function.
- Optimizer: A specific implementation of the gradient descent algorithm. (There are many algorithms for this. In this course we will only use the “Adam” Optimizer, which stands for - ADAptive with Momentum. It is considered the best-practice optimizer.)
- Learning rate: The “step size” for loss improvement during gradient descent.
- Batch: The set of examples used during training of the neural network
- Epoch: A full pass over the entire training dataset
- Forward pass: The computation of output values from input
- Backward pass (backpropagation): The calculation of internal variable adjustments according to the optimizer algorithm, starting from the output layer and working back through each layer to the input.
>  *Training Set: The data used for training the neural network.
>  *Test set: The data used for testing the final performance of our neural network. (to try the network on data it has never seen before)
>  *The validation set is used again when training is complete to measure the final accuracy of the model.

- CNNs: Convolutional neural network. That is, a network which has at least one convolutional layer. A typical CNN also includes other types of layers, such as pooling layers and dense layers
- Kernel / filter: A matrix which is smaller than the input, used to transform the input into chunks
- Convolution: the process of applying a filter (“kernel”) to an image. 
- Downsampling: The act of reducing the size of an image
-  Max pooling: the process of reducing the size of the image through downsampling.
-  Stride: the number of pixels to slide the kernel (filter) across the image.
- Padding: Adding pixels of some value, usually 0, around the input image
- Pooling The process of reducing the size of an image through downsampling.There are several types of pooling layers. For example, average pooling converts many values into a single value by taking the average. However, maxpooling is the most common.

# To Prevent Overfitting
- Early Stopping: In this method, we track the loss on the validation set during the training phase and use it to determine when to stop training such that the model is accurate but not overfitting.
- Image Augmentation: Artificially boosting the number of images in our training set by applying random image transformations to the existing images in the training set.
- Dropout: Removing a random selection of a fixed number of neurons in a neural network during training.


