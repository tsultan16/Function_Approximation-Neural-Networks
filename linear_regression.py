'''
   ##########################################
   # Linear Regression via Gradient Descent #
   ##########################################

    In this problem, we have a quantity 'y' whose value depends on some features (i.e. independant variables) 'x = (x_1, x_2, x_3, ..., x_N)'

    We define a quantity 'p' that is an approximation to 'y' and hypothesize that there is a linear relationship between p and x, i.e.

        p(x) = w_0 + w_1*x_1 + w_2*x_2 + w_3*x_3 + ... + x_N*x_N

    where w = (w_i, i = 0, 2, 3.., N) are constant 'weights'. 

    Now, given a data sample of observations of y values and corresponding x values of each:

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

    This loss function L is a function of the weights. Smaller the loss, greater the accuracy of our approximation. The goal then is to find a combination of weights that will minimize the loss function, i.e. dL/dw (w_min) = 0. (This could be either a minima or a maxima, but that won't be an issue because we'll use gradient descent). To find the (local or glabal) minimum of L(w), we can use the 'gradient descent approach'. I.e. we start with some arbitrary initial value of the weights, w_* = (w_0_8, w_1_*, ...,w_N_*), then evaluate the loss function gradient at this value, dL/dw (w_*), then we translate the weights values in the direction of the minima (i.e. in the direction of downward slope, or negative gradient) by a small amount proportional to the gradient :
    
        w_*_updated = w_*_old - a * dL/dw (w_*_old)

    where 'a' is a porportionality constant. We want 'a' to be small enough so that we don't overshoot the loss function minima. We itertate this pocess of upodating the weights and re-evauating the loss fcuntion gradient until we reach a point where the gradient value becomes sufficiently close to zero indicating that we're near the minima. Then we copute the approximate values p using these latest weights and we're done!        

'''


import numpy as np
from typing import Callable, Dict, Tuple, List
import matplotlib.pyplot as plt

def forward_loss(X_batch: np.ndarray, y_batch: np.ndarray, weights: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, np.ndarray]]:

    # make sure batch sizes are equal
    assert X_batch.shape[0] == y_batch.shape[0], "Batch sizes do not match" 

    # make sure matrix multiplication is possible
    assert X_batch.shape[1] == weights['W'].shape[0], "Matrix shapes don't allow multiplication"

    # make sure that B is a 1x1 array
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1, "B needs to be a 1x1 array"

    # dot product of weights with X
    N = np.dot(X_batch, weights['W'])

    # compute predictions
    P = N + weights['B']

    # compute loss
    loss = np.mean(np.power(y_batch-P, 2))  

    # save information computed on this forward pass
    forward_info: Dict[str, np.ndarray] = {}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['y'] = y_batch

    return forward_info, loss

def loss_gradients(forward_info: Dict[str, np.ndarray], weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    
    batch_size = forward_info['X'].shape[0]

    # dL/dP
    dL_dP = -2 * (forward_info['y']- forward_info['P'])

    # dP/dN
    dP_dN = np.ones_like(forward_info['N'])
 
    # dP/dB
    dP_dB = np.ones_like(weights['B'])

    # dL/dN
    dL_dN = dL_dP * dP_dN

    # dN/dW
    dN_dW = np.transpose(forward_info['X'],(1,0))

    # dL/dW (derivative of loss function w.r.t weights)
    dL_dW = np.dot(dN_dW, dL_dN)

    # dL/dB
    dL_dB = (dL_dP * dP_dB).sum(axis = 0)


    loss_gradients : Dict[str, np.ndarray] = {}
    loss_gradients['W'] = dL_dW
    loss_gradients['B'] = dL_dB

    return loss_gradients

Batch = Tuple[np.ndarray, np.ndarray]

def generate_batch(X: np.ndarray, y: np.ndarray, start: int = 0, batch_size: int = 10) -> Batch:

    assert X.ndim == y.ndim == 2, "X and y need to be 2d arrays"

    # generate batch from X and y, given the starting position
    if start + batch_size > X.shape[0]:
        batch_size = X.shape[0] - start

    X_batch = X[start:start+batch_size]    
    y_batch = y[start:start+batch_size]    

    return X_batch, y_batch

def init_weights(n_in: int) -> Dict[str, np.ndarray]: 

    # initialize the weights
    weights: Dict[str, np.ndarray] = {}

    weights['W'] = np.random.randn(n_in, 1)
    weights['B'] = np.random.randn(1, 1)

    return weights

def train(X: np.ndarray, y: np.ndarray, n_iter: int = 1000, learning_rate: float = 0.01, 
          batch_size: int = 100, return_losses: bool = False, return_weights: bool = False,
          seed: int = 1) -> None:

    print("Commencing training...")

    if seed: 
        np.random.seed(seed)

    start = 0

    # initialize weights
    weights = init_weights(X.shape[1]) 

    # permute data
    X, y = permute_data(X, y)

    #if (return_losses):
    losses = []

    for i in range(n_iter):
        
        # generate batch
        if (start >= X.shape[0]):
            X, y = permute_data(X, y)
            start = 0

        X_batch, y_batch = generate_batch(X, y, start, batch_size)
        start += batch_size

        # train net using generated batch
        forward_info, loss = forward_loss(X_batch, y_batch, weights)    

        if(return_losses):
            losses.append(loss)

        loss_grads = loss_gradients(forward_info, weights)

        # update the weights
        weights['W'] -= learning_rate * loss_grads['W']     
        weights['B'] -= learning_rate * loss_grads['B']     

    print("Training complete!")
    print("P = ")
    print(forward_info['P'])

    if (return_weights):
        return losses, weights
    else:
        return losses, []

    return None

def permute_data(X: np.ndarray, y: np.ndarray):
    # randomly permute the X and y arrays along axis = 0 (rows)
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

######################################################################################


X = np.random.randn(4,4)
y = np.random.randn(4,1)

print("X = ")
print(X)
print("y = ")
print(y)

train_info = train(X, y, n_iter = 1000, learning_rate = 0.001, 
          batch_size = 23, return_losses = True, return_weights = True,
          seed = 182635)

losses = train_info[0]
weights = train_info[1]


plt.plot(np.arange(1000), losses)
plt.show()