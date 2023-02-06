'''
   ##########################################
   # Linear Regression via Gradient Descent #
   ##########################################

    In this problem, we have a quantity 'y' whose value depends on N different features (i.e. independant variables) 'x = (x_1, x_2, x_3, ..., x_N)'

    We hypothesize that there is a linear relationship between y and x, and define a quantity 'p' which is an approximation to 'y':

        p(x) = w_0 + w_1*x_1 + w_2*x_2 + w_3*x_3 + ... + x_N*x_N

    where w = (w_i, i = 0, 1, 2, 3.., N) are constant 'weights'. 

    Now, given a data sample containing 'K' differenct observations of y and corresponding x =(x_k_1, x_k_2, ...,x_k_N) of each:

        | y_1 |       | x_1_1, x_1_2, ..., x_1_N | 
        | y_2 |       | x_2_1, x_2_2, ..., x_2_N |
        | y_3 |       | x_3_1, x_3_2, ..., x_3_N |
        | y_4 |  <->  | x_4_1, x_4_2, ..., x_4_N |
        | :   |       |   :                      |
        | :   |       |   :                      |
        | y_K |       | x_K_1, x_K_2, ..., x_K_N |
    
    
    we can use our hypothesis function p(x) to compute approximations for each of these y value observations as follows:

        | p_1 |      | w_0 |     | x_1_1  x_1_2  .... x_1_N |      | w_1 |
        | p_2 |      | w_0 |     | x_2_1  x_2_2  .... x_2_N |      | w_2 |
        | p_3 |      | w_0 |     | x_3_1  x_3_2  .... x_3_N |      | w_3 |       
        | p_4 |   =  | w_0 |  +  | x_4_1  x_4_2  .... x_4_N |  *   | w_4 |
        | :   |      | :   |     |   :      :           :   |      |  :  |
        | :   |      | :   |     |   :      :           :   |      |  :  |
        | p_K |      | w_0 |     | x_K_1  x_K_2  .... x_K_N |      | w_N |
    

    the second term on right hand side is a matrix multiplication.
    
    We then define an error/loss function to quantify the accuracy of this approximation (in this case, we choose leasts squares error):

        L(w) =  (1/N) * sum_i =[1 to N] (p_i - y_i)^2 

    This loss function L is a function of the weights. Smaller the loss, greater the accuracy of our approximation. The goal then is to
    find a combination of weights that will minimize the loss function, i.e. dL/dw (w_min) = 0. (This could be either a minima or a
    maxima, but that won't be an issue because we'll use gradient descent). To find the (local or glabal) minimum of L(w), we can use the
    'gradient descent approach'. I.e. we start with some arbitrary initial value of the weights, w_* = (w_0_*, w_1_*, ...,w_N_*), then
    evaluate the loss function gradient at this value, dL/dw (w_*), then we translate the weights values in the direction of the minima
    (i.e. in the direction of downward slope, or negative gradient) by a small amount proportional to the gradient :
    
        w_*_updated = w_*_old - a * dL/dw (w_*_old)

    where 'a' is a porportionality constant (also referred to as the "learning rate"). We want 'a' to be small enough so that we don't 
    overshoot the loss function minima. We iterate this pocess of re-evauating the loss funtion gradient and updating the weights 
    until we reach a point where the gradient value becomes sufficiently close to zero indicating that we're near the minima. Then we
    compute the approximate values p using these latest weights and we're done!        

'''

import math
import numpy as np
from typing import Callable, Dict, Tuple, List
import matplotlib.pyplot as plt

def compute_loss(X: np.ndarray, y: np.ndarray, weights: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, np.ndarray]]:

    # make sure batch sizes are equal
    assert X.shape[0] == y.shape[0], "Batch sizes do not match" 

    # make sure matrix multiplication is possible
    assert X.shape[1] == weights['W'].shape[0], "Matrix shapes don't allow multiplication"

    # make sure that B is a 1x1 array
    assert weights['W0'].shape[0] == weights['W0'].shape[1] == 1, "W0 needs to be a 1x1 array"

    # dot product of weights with X
    N = np.dot(X, weights['W'])

    # compute predictions
    P = N + weights['W0']

    # compute loss function
    loss = np.mean(np.power(y-P, 2))  

    # save information computed on this forward pass
    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y

    return forward_info, loss

def compute_loss_gradient(forward_info: Dict[str, np.ndarray], weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    
    # dL/dP 
    dL_dP = -(2/forward_info['y'].shape[0]) * (forward_info['y']- forward_info['P'])

    # dP/dN
    # dP_dN = np.ones_like(forward_info['N'])
 
    # dP/dW0
    #dP_dW0 = np.ones_like(weights['W0'])

    # dL/dN
    #dL_dN = dL_dP * dP_dN
    dL_dN = dL_dP

    # dN/dW
    dN_dW = np.transpose(forward_info['X'],(1,0))

    # dL/dW (derivative of loss function w.r.t weights)
    dL_dW = np.dot(dN_dW, dL_dN)

    # dL/dW0
    #dL_dW0 = (dL_dP * dP_dW0).sum(axis = 0)
    dL_dW0 = dL_dP.sum(axis = 0)

    loss_gradient : Dict[str, np.ndarray] = {}
    loss_gradient['W'] = dL_dW
    loss_gradient['W0'] = dL_dW0

    return loss_gradient

def init_weights(N: int) -> Dict[str, np.ndarray]: 

    # initialize the weights
    weights: Dict[str, np.ndarray] = {}

    weights['W'] = np.random.randn(N, 1)
    weights['W0'] = np.random.randn(1, 1)

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
          batch_size: int = 10,
          seed: int = 1):

    print("Commencing training...")

    if seed: 
        np.random.seed(seed)

    # initialize weights
    weights = init_weights(X.shape[1]) 
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
niterations = 500 # number of gradient descent interations

# generate some test sample data
X = np.random.randn(K,N)
#y = np.random.randn(K,1)
perfect_weights = init_weights(N) 
y = perfect_weights['W0'] + np.dot(X, perfect_weights['W'])
for i in range(y.shape[0]):
    y[i,0] *= 1 + (np.random.uniform(0,1,1)[0] - 0.5) * 0.5

#print("X = ")
#print(X)
#print("y = ")
#print(y)

# train the approximation
train_info = train(X, y, n_iter = niterations, learning_rate = 0.001, 
          batch_size = 50,
          seed = 182635)

losses = train_info[0]
weights = train_info[1]

P = weights['W0'] + np.dot(X, weights['W'])

max_y = np.amax(y)
min_y = np.amin(y)
max_P = np.amax(P)
min_P = np.amin(P)

z = np.linspace(math.floor(min(min_y, min_P)), math.ceil(max(max_y, max_P)), 20*N)

print("max_y = ",max_y)
print("max_P = ",max_P)

plt.subplot(1,2,1)
plt.plot(np.arange(len(losses)), losses)
plt.subplot(1,2,2)
plt.scatter(P, y, s=2)
plt.plot(z, z, 'r--')
plt.show()