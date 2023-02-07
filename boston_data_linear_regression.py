from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from linear_regression import *

#######################
# load boston dataset #
#######################

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

K = y_test.shape[0]  # number of test observations
N = X_test.shape[1]  # number of features
niterations = 500    # number of gradient descent interations

# train the dataset and generate weights
train_info = train(X_train, y_train, n_iter = niterations, learning_rate = 0.001, batch_size = 50, seed = 182635)

losses = train_info[0]
weights = train_info[1]

P_test = predict(X_test, weights)

max_y = np.amax(y_test)
min_y = np.amin(y_test)
max_P = np.amax(P_test)
min_P = np.amin(P_test)

z = np.linspace(math.floor(min(min_y, min_P)), math.ceil(max(max_y, max_P)), 20*N)


# Dependance of target 'y' on it's most important feature, i.e. feature corresponding to the weight with the highest magnitude

i_mi = np.argmax(np.abs(weights['W']))
#print("Most-important feature index: ", i_mi)
#print("Value of most important weight: ",weights['W'][i_mi,0])
mean_X = (1.0 / K) * X_test.sum(axis=0)
new_X = np.zeros((20*K,N)) + mean_X
x_mi = X_test[:,i_mi]
x_p = np.linspace(np.amin(x_mi), np.amax(x_mi), 20*K)
new_X[:,i_mi] = x_p
y_p = predict(new_X, weights) #weights['W0'] + np.dot(new_X, weights['W'])

plt.subplot(1,3,1)
plt.plot(np.arange(len(losses)), losses)
plt.xlabel('# of iterations')
plt.ylabel('loss/error')
plt.subplot(1,3,2)
plt.scatter(P_test, y_test, s=4)
plt.plot(z, z, 'r--')
plt.xlabel('P')
plt.ylabel('y')
plt.subplot(1,3,3)
plt.scatter(x_mi, y_test, s=4)
plt.plot(x_p, y_p, 'r--', linewidth=1)
plt.xlabel('x_mi')
plt.ylabel('y')

plt.show()
