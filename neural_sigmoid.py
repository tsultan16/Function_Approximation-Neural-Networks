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
    M2 = np.dot(N1, weights['W2'])
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
    dL_dB2 = -(2/forward_info['y'].shape[0]) * np.sum(yminusP) * np.ones_like(forward_info['B2'])

    # dL/dW1
    Del = forward_info['O1'] * (1 - forward_info['O1'])
    W2_T = np.transpose(weights['W2'],(1,0))
    R = np.dot(yminusP, W2_T) * Del 
    X_T = np.transpose(forward_info('X'),(1,0))
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

def init_weights(N: int, M: int) -> Dict[str, np.ndarray]: 

    # initialize the weights
    weights: Dict[str, np.ndarray] = {}

    weights['W1'] = np.random.randn(N, M)
    weights['B1'] = np.random.randn(1, M)
    weights['W2'] = np.random.randn(M, 1)
    weights['B2'] = np.random.randn(1, 1)

    return weights

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
          hidden_layers: int = 5, batch_size: int = 10,
          seed: int = 1):

    print("Commencing training...")

    if seed: 
        np.random.seed(seed)

    # initialize weights
    weights = init_weights(X.shape[1], hidden_layers) 
    losses = []

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

            # update the weights (gradient descent)
            weights['W']  -= learning_rate * loss_grads['W']     
            weights['W0'] -= learning_rate * loss_grads['W0']     

        losses.append(total_loss)


    print("Training complete!")

    return losses, weights

######################################################################################

K = 1000 # number of observations
N = 50  # number of features
M = 10 # number of hidden layers
niterations = 500 # number of gradient descent interations

# generate some test sample data
X = np.random.randn(K,N)

# linear y
#test_weights = init_weights(N) 
#y = test_weights['W0'] + np.dot(X, test_weights['W'])

# add some small random deviations from linearity
#for i in range(y.shape[0]):
#    y[i,0] *= 1 + (np.random.uniform(0,1,1)[0] - 0.5) * 0.5  


# train the approximation
train_info = train(X, y, n_iter = niterations, hidden_layers = M, learning_rate = 0.001,batch_size = 50, seed = 182635)

losses = train_info[0]
weights = train_info[1]
