# TwoDigitMNIST
Classification of 2-digit numbers formed from MNIST dataset using 2-hidden layer neural network built using PyTorch.
The dataset is formed by taking a batch of 10 random images from Original MNIST and concatenating every image with every other image. This way we have dataset with size 10 times the original and labels also 10 times what the original dataset had.

## Data Preparation

I built a function ```pari()``` that takes a batch of 10 MNIST images and 10 labels and returns concatenated images and labels. I used this function during the training and testing of the model.

## Model Building and Training

I built the neural network on top of ```torch.nn.Module```. My model has 2 hidden layers with ```ReLU``` activation and 250, 250, 200 neurons each. I have used Stochastic Gradient Descent to train the model. The train and test functions print out the respective accuracies each epoch.

## Model Performance

The model takes around 6 minutes overall to run on my machine on ```cuda``` device. The train set was split 90-10 into train and validation and the validation accuracy was 97% while the actual test accuracy was 96%.

## Scope of Improvement

We have multiple points that can be improved upon. We can built a custom dataset instead of using the ```pair()``` function. This would improve on the speed. We could also experiment with alternate optimizers.
