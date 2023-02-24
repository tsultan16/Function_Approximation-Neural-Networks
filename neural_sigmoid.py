import math
import numpy as np
from typing import Callable, Dict, Tuple, List
import matplotlib.pyplot as plt

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))

def compute_loss(X: np.ndarray, y: np.ndarray, weights: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, np.ndarray]]:

    # make sure batch sizes are equal
    assert X.shape[0] == y.shape[0], "Batch sizes do not match" 

    # make sure matrix multiplication is possible
    assert X.shape[1] == weights['W1'].shape[0], "Matrix shapes don't allow multiplication"

    # dot product of weights with X
    M1 = np.dot(X, weights['W1'])
    N1 = weights['B1'] + M1

    # apply sigmoid function
    O1 = sigmoid(N1)

    # compute predictions
    M2 = np.dot(O1, weights['W2'])
    P = weights['B2'] + M2

    # compute loss function
    loss = np.mean(np.power(y-P, 2))  

    # save information computed on this forward pass
    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X
    forward_info['M1'] = M1
    forward_info['N1'] = N1
    forward_info['M2'] = M2
    forward_info['O1'] = O1
    forward_info['P'] = P
    forward_info['y'] = y

    return forward_info, loss

def compute_loss_gradient(forward_info: Dict[str, np.ndarray], weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    
    # dL/dW2 
    O1_T = np.transpose(forward_info['O1'],(1,0))
    yminusP = forward_info['y']- forward_info['P']
    dL_dW2 = -(2/forward_info['y'].shape[0]) * np.dot(O1_T, yminusP)

    # dL/dB2
    dL_dB2 = -(2/forward_info['y'].shape[0]) * np.sum(yminusP) * np.ones_like(weights['B2'])

    # dL/dW1
    Del = forward_info['O1'] * (1 - forward_info['O1'])
    W2_T = np.transpose(weights['W2'],(1,0))
    R = np.dot(yminusP, W2_T) * Del 
    X_T = np.transpose(forward_info['X'],(1,0))
    dL_dW1 = -(2/forward_info['y'].shape[0]) * np.dot(X_T, R) 

    # dL/dB1
    yminusP_T = np.transpose(yminusP,(1,0))
    dL_dB1 = -(2/forward_info['y'].shape[0]) * W2_T * np.dot(yminusP_T, Del)

    loss_gradient : Dict[str, np.ndarray] = {}
    loss_gradient['W1'] = dL_dW1
    loss_gradient['W2'] = dL_dW2
    loss_gradient['B1'] = dL_dB1
    loss_gradient['B2'] = dL_dB2

    return loss_gradient

def predict(X: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    
    # make sure matrix multiplication is possible
    assert X.shape[1] == weights['W1'].shape[0], "Matrix shapes don't allow multiplication"

    # dot product of weights with X
    M1 = np.dot(X, weights['W1'])
    # add bias term
    N1 = weights['B1'] + M1

    # apply sigmoid function to get output of first layer
    O1 = sigmoid(N1)
    
    # compute predictions (i.e. output of second layer)
    M2 = np.dot(O1, weights['W2'])
    P = weights['B2'] + M2
    return P

def mean_absolute_error(P: np.ndarray, y: np.ndarray) -> float:
    return np.mean(np.abs(y-P))    

def root_mean_squared_error(P: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.mean(np.power(y-P, 2)))    

def init_weights(N: int, M: int) -> Dict[str, np.ndarray]: 

    # initialize the weights
    weights: Dict[str, np.ndarray] = {}

    weights['W1'] = np.random.randn(N, M)
    weights['B1'] = np.random.randn(1, M)
    weights['W2'] = np.random.randn(M, 1)
    weights['B2'] = np.random.randn(1, 1)

    return weights

def init_velocities(N: int, M: int) -> Dict[str, np.ndarray]: 

    # initialize the weights
    velocities: Dict[str, np.ndarray] = {}

    velocities['W1'] = np.zeros(shape=(N, M))
    velocities['B1'] = np.zeros(shape=(1, M))
    velocities['W2'] = np.zeros(shape=(M, 1))
    velocities['B2'] = np.zeros(shape=(1, 1))

    return velocities


Batch = Tuple[np.ndarray, np.ndarray]
def generate_batch(X: np.ndarray, y: np.ndarray, batch_lo: int = 0, batch_size: int = 10) -> Batch:

    assert X.ndim == y.ndim == 2, "X and y need to be 2d arrays"

    # generate batch from X and y, given the starting position
    if batch_lo + batch_size > X.shape[0]:
        batch_size = X.shape[0] - batch_lo

    X_batch = X[batch_lo:batch_lo+batch_size]    
    y_batch = y[batch_lo:batch_lo+batch_size]    

    return X_batch, y_batch

def train(X: np.ndarray, y: np.ndarray, n_iter: int = 1000, learning_rate: float = 0.01, 
          momentum: float = 0.9, n_hidden_weights: int = 5, batch_size: int = 10,
          seed: int = 1):

    print("Commencing training...")

    if seed: 
        np.random.seed(seed)

    # initialize weights
    weights = init_weights(X.shape[1], n_hidden_weights) 

    # initialize velocities
    velocities = init_velocities(X.shape[1], n_hidden_weights)
    
    losses = []
    mabs_err = []
    rms_err = []
    scaled_rms_err = []
    ymean = np.mean(y)

    #number of batches
    n_batch = math.ceil(X.shape[0]/batch_size)

    print("Number of batches = ",n_batch)

    # gradient descent iterations
    for i in range(n_iter):
        
        # For data sets containing a large number of observations (i.e. K is large), we will 
        # work in batches (to avoid memory overflow from matrix multiplications that are too large)
        batch_lo = 0
        total_loss = 0

        # iterating over batches
        for j in range(n_batch):
            
            # generate batch
            X_batch, y_batch = generate_batch(X, y, batch_lo, batch_size)
            batch_lo += batch_size

            # train using generated batch
            forward_info, loss = compute_loss(X_batch, y_batch, weights)    
            total_loss += loss
            loss_grads = compute_loss_gradient(forward_info, weights)

            '''
            # update the weights (gradient descent without momentum)
            for key in weights.keys():
                update = learning_rate * loss_grads[key]
                weights[key] -= update   
            '''

            # update the weights and velocities (gradient descent with momentum)
            for key in weights.keys():
                update = learning_rate * loss_grads[key] + momentum * velocities[key]
                weights[key] -= update
                velocities[key] = update
 
        losses.append(total_loss)
        print(f"Loss after iteration#{i+1} is {total_loss}")

        # compute the prediciton and associated errors
        P = predict(X, weights)
        mabs_err.append(mean_absolute_error(P,y))  
        rms =  root_mean_squared_error(P,y)
        rms_err.append(rms)   
        scaled_rms_err.append(rms/ymean)   

    print("Training complete!")

    return losses, weights, mabs_err, rms_err, scaled_rms_err

######################################################################################
'''
K = 1000 # number of observations
N = 50  # number of features
M = 5 # number of hidden layers
niterations = 2000 # number of gradient descent interations

# generate some test sample data
X = np.random.randn(K,N)

# perfect y
test_weights = init_weights(N,M) 
y = predict(X, test_weights)

# add some small random deviations
#for i in range(y.shape[0]):
#    y[i,0] *= 1 + (np.random.uniform(0,1,1)[0] - 0.5) * 0.1  


# train the approximation
train_info = train(X, y, n_iter = niterations, n_hidden_weights = M, learning_rate = 0.001,batch_size = 50, seed = 182635)

losses = train_info[0]
weights = train_info[1]
mabs_err = train_info[2]
rms_err = train_info[3]
scaled_rms_err = train_info[4]

P = predict(X, weights)
max_y = np.amax(y)
min_y = np.amin(y)
max_P = np.amax(P)
min_P = np.amin(P)
z = np.linspace(math.floor(min(min_y, min_P)), math.ceil(max(max_y, max_P)), 20*N)


plt.subplot(2,2,1)
plt.plot(np.arange(len(losses)), losses)
plt.xlabel('# of iterations')
plt.ylabel('loss')
plt.subplot(2,2,2)
plt.scatter(P, y, s=2)
plt.plot(z, z, 'r--')
plt.xlabel('P')
plt.ylabel('y')
plt.subplot(2,2,3)
plt.plot(np.arange(len(mabs_err)), mabs_err)
plt.xlabel('# of iterations')
plt.ylabel('means_abs_error')
plt.subplot(2,2,4)
plt.plot(np.arange(len(scaled_rms_err)), scaled_rms_err)
plt.xlabel('# of iterations')
plt.ylabel('rms_error/y_mean)')

plt.show()
'''