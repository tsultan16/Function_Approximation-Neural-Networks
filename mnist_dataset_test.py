from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from neural_network_mod import *


'''
    MNIST dataset of handwritten digits:

    Each observation is an image. The input values per image are 28x28 pixels (i.e. 784 features/inputs per observation).
'''


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# rescaling data to have mean = 0 and variance = 1 (test data normalized using training data)
X_train, X_test = X_train - np.mean(X_train), X_test - np.mean(X_train) 
X_train, X_test = X_train/np.std(X_train) , X_test/np.std(X_train) 
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print(f"X_train shape = {X_train.shape}")
print(f"y_train shape = {y_train.shape}")
print(f"X_test shape = {X_test.shape}")
print(f"y_test shape = {y_test.shape}")

# flatten 28x28 inputs into 1d
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

print("Post flattening:")
print(f"X_train shape = {X_train.shape}")
print(f"y_train shape = {y_train.shape}")
print(f"X_test shape = {X_test.shape}")
print(f"y_test shape = {y_test.shape}")

'''
# visualizing some of the images (sanity check)
plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i])
    plt.title(f"Label = {y_train[i]}")
plt.show()
'''

'''
    Deep learning model for classifying the image dataset:

    In this dataset, there are then classes for the target, i.e. the digits (0,1,2,3,4,5,6,7,8,9)
    The output from our neural network corresponding to each observation will contain an array of ten probabilities, one for each class, e.g. if the probability for class=2 is 0.99 => prediction of our neural network is that the input image belongs to class=2. 

    We will use a model with two layers, we will use a tanh activation function at the end of the first layer and sigmoid at the end of the second layer since the outputs represent probabilities for each class and should have values in the range (0,1).

    We will set the number of neurons in the hidden layers to the geometric mean of the number of inputs and the number of outputs per image, i.e. sqrt(784 * 10) ~ 89  

 '''

# transform target array: convert each class label into array of probabilities (1 for a single class, zero for the other nine)
# e.g. class = '2' will be represented by the array [0 0 1 0 0 0 0 0 0 0]
#      class = '7' => [0 0 0 0 0 0 0 1 0 0]

y_train_tr = np.zeros(shape = (y_train.shape[0], 10)) 
y_test_tr = np.zeros(shape = (y_test.shape[0], 10)) 
for i in range(y_train.shape[0]):
    y_train_tr[i, y_train[i,0]] = 1
for i in range(y_test.shape[0]):
    y_test_tr[i, y_test[i,0]] = 1


model = NeuralNetwork(layers = [Dense(neurons = 89, activation = Tanh()), Dense(neurons = 10, activation = Sigmoid())], loss = MeanSquaredError(), seed = 20190119) 

optimizer = SGD(lr = 0.1)
trainer = Trainer(model, optimizer)

trainer.fit(X_train, y_train_tr, X_test, y_test_tr, epochs = 50, eval_every = 10, batch_size = 60, seed = 20190119)
P =  model.forward(X_test)

for i in range(10):
    print(f"target: {y_test[i]}, predictions: {P[i]}")
    print("")


