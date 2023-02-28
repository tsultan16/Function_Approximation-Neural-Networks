from neural_network_mod_book import *
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math

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
deep_learning_network = NeuralNetwork(layers = [Dense(neurons = 13, activation = Sigmoid()), Dense(neurons = 13, activation = Sigmoid()), Dense(neurons = 1, activation = Linear())], loss = MeanSquaredError(), seed = 123456) 

# create optimizer and trainer instances and train the dataset
#optimizer = SGD(lr = 0.01)

trainer = Trainer(linear_regression, SGD(lr = 0.01))
trainer.fit(X_train, y_train, X_test, y_test, epochs = 50, eval_every = 10, seed = 20190501)
regression_model_errors(linear_regression, X_test, y_test)
print("#"*80)
trainer = Trainer(sigmoid_neural_network, SGD(lr = 0.01))
trainer.fit(X_train, y_train, X_test, y_test, epochs = 50, eval_every = 10, seed = 20190501)
regression_model_errors(sigmoid_neural_network, X_test, y_test)
print("#"*80)
trainer = Trainer(deep_learning_network, SGD(lr = 0.01))
trainer.fit(X_train, y_train, X_test, y_test, epochs = 50, eval_every = 10, seed = 20190501)
regression_model_errors(deep_learning_network, X_test, y_test)

