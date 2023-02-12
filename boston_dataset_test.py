from neural_network_mod import *
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt

# Test datasets:

##################
# linear dataset #
##################
'''
N = 20 # number of features
K = 500 # number of observations

X_train = np.random.randn(K,N) 
X_test = np.random.randn(K,N) 

# linear target
test_weights = np.random.randn(N,1) 
test_bias = np.random.randn(1,1)
y_train = test_bias + np.dot(X_train, test_weights)
y_test = test_bias + np.dot(X_test, test_weights)


#print("X_train = ")
#print(X_train)
#print("y_train")
#print(y_train)
'''

##################
# boston dataset #
##################

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names
s = StandardScaler()
data = s.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)


print("X_train shape = ",X_train.shape, ", y_train shape = ",y_train.shape)
print("X_test shape = ",X_test.shape, ", y_test shape = ",y_test.shape)



#############################################
# train the dataset and generate prediction #
#############################################


# create instances of linear regression and one-layer sigmoid neural networks
linear_regression      = NeuralNetwork(layers = [Dense(neurons = 1, activation = Linear())], loss = MeanSquaredError(), seed = 123456)
sigmoid_neural_network = NeuralNetwork(layers = [Dense(neurons = 13, activation = Sigmoid()), Dense(neurons = 1, activation = Linear())], loss = MeanSquaredError(), seed = 123456) 
sigmoid_neural_network_tanh = NeuralNetwork(layers = [Dense(neurons = 13, activation = Tanh()), Dense(neurons = 1, activation = Linear())], loss = MeanSquaredError(), seed = 123456) 
deep_learning_network = NeuralNetwork(layers = [Dense(neurons = 13, activation = Sigmoid()), Dense(neurons = 13, activation = Sigmoid()), Dense(neurons = 1, activation = Linear())], loss = MeanSquaredError(), seed = 123456) 
deep_learning_network_tanh = NeuralNetwork(layers = [Dense(neurons = 13, activation = Tanh()), Dense(neurons = 13, activation = Tanh()), Dense(neurons = 1, activation = Linear())], loss = MeanSquaredError(), seed = 123456) 

# create optimizer and trainer instances and train the dataset
#optimizer = SGD(lr = 0.01)

trainer = Trainer(linear_regression, SGD(lr = 0.01))
trainer.fit(X_train, y_train, X_test, y_test, epochs = 50, eval_every = 10, seed = 20190501)
regression_model_errors(linear_regression, X_test, y_test)
P_linear =  linear_regression.forward(X_test) 
print("#"*80)
trainer = Trainer(sigmoid_neural_network, SGD(lr = 0.01))
trainer.fit(X_train, y_train, X_test, y_test, epochs = 50, eval_every = 10, seed = 20190501)
P_sigmoid =  sigmoid_neural_network.forward(X_test)
regression_model_errors(sigmoid_neural_network, X_test, y_test)
print("#"*80)
trainer = Trainer(deep_learning_network, SGD(lr = 0.01))
trainer.fit(X_train, y_train, X_test, y_test, epochs = 50, eval_every = 10, seed = 20190501)
P_deep =  deep_learning_network.forward(X_test)
regression_model_errors(deep_learning_network, X_test, y_test)
print("#"*80)
trainer = Trainer(sigmoid_neural_network_tanh, SGD(lr = 0.01))
trainer.fit(X_train, y_train, X_test, y_test, epochs = 50, eval_every = 10, seed = 20190501)
P_sigmoid =  sigmoid_neural_network_tanh.forward(X_test)
regression_model_errors(sigmoid_neural_network_tanh, X_test, y_test)
print("#"*80)
trainer = Trainer(deep_learning_network_tanh, SGD(lr = 0.01))
trainer.fit(X_train, y_train, X_test, y_test, epochs = 50, eval_every = 10, seed = 20190501)
P_deep =  deep_learning_network_tanh.forward(X_test)
regression_model_errors(deep_learning_network_tanh, X_test, y_test)



###############################################################################################
make_plots = False

if (make_plots):

    K = y_test.shape[0]  # number of test observations
    N = X_test.shape[1]  # number of features
    M = 13 # number neurons

    max_y = np.amax(y_test)
    min_y = np.amin(y_test)
    max_P = np.amax(P_linear)
    min_P = np.amin(P_linear)
    z = np.linspace(math.floor(min(min_y, min_P)), math.ceil(max(max_y, max_P)), 20*N)

    # Dependance of target 'y' on it's most important feature, i.e. feature corresponding to the weight with the highest magnitude
    i_mi = 12 #np.argmax(np.abs(weights['W']))
    x_mi = X_test[:,i_mi]

    plt.subplot(2,3,1)
    plt.scatter(P_linear, y_test, s=4)
    plt.plot(z, z, 'r--')
    plt.xlabel('P')
    plt.ylabel('y')
    plt.title("Linear")

    plt.subplot(2,3,2)
    plt.scatter(P_sigmoid, y_test, s=4)
    plt.plot(z, z, 'r--')
    plt.xlabel('P')
    plt.ylabel('y')
    plt.title("Sigmoid")

    plt.subplot(2,3,3)
    plt.scatter(P_deep, y_test, s=4)
    plt.plot(z, z, 'r--')
    plt.xlabel('P')
    plt.ylabel('y')
    plt.title("Deep")

    plt.subplot(2,3,4)
    plt.scatter(x_mi, y_test, s=4)
    plt.scatter(x_mi, P_linear, c ='red', s=4)
    plt.xlabel('x_mi')
    plt.ylabel('y')
    plt.title("Linear")

    plt.subplot(2,3,5)
    plt.scatter(x_mi, y_test, s=4)
    plt.scatter(x_mi, P_sigmoid, c ='red', s=4)
    plt.xlabel('x_mi')
    plt.ylabel('y')
    plt.title("Sigmoid")

    plt.subplot(2,3,6)
    plt.scatter(x_mi, y_test, s=4)
    plt.scatter(x_mi, P_deep, c ='red', s=4)
    plt.xlabel('x_mi')
    plt.ylabel('y')
    plt.title("Deep")

    plt.show()