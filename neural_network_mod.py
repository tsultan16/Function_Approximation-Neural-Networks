'''
    A self-contained multi-layer neural network module
'''

import numpy as np
from scipy.special import logsumexp
from typing import Callable, Dict, Tuple, List
import copy

'''
function that checks if two arrays have atching shape
'''
def assert_same_shape(array1: np.ndarray, array2: np.ndarray) -> None:
    assert array1.shape == array2.shape, "Array shapes do not match"
    return None

'''
function for shuffling the rows of the given arrays
'''
def permute_data(X, y):

    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


'''
Softmax function
'''
def softmax(P: np.ndarray, sf_axis: int):

    # using logsumexp instead of directly computing exp(P) to avoid numerical overflow
    S = logsumexp(P, axis = sf_axis)
    S = np.reshape(S, (P.shape[0], 1))
    softmax_P = np.exp(P-S) 
    return softmax_P

'''
A generalised base class for neural network operations
All operations will inherit from this base class
'''
class Operation(object):
    '''
    constructor
    '''
    def __init__(self):
        pass

    '''
    forward pass: computes the result of performing the opertion on the inputs
    '''
    def forward(self, input_: np.ndarray) -> np.ndarray :
        # store the input_ array in the self.input_ instance variable
        self.input_ = input_

        # compute and return the output
        self.output = self._output()
        return self.output

    '''
    backward pass: computes the gradients of the operation output w.r.t. the inputs
    '''
    def backward(self, output_grad: np.ndarray) -> np.ndarray:   
        # make sure output gradient arrays and output are the same shape 
        assert_same_shape(self.output, output_grad) 

        # compute and return input gradients
        self.input_grad = self._input_grad(output_grad)

        # make sure input gradient arrays are same shape as inputs 
        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> np.ndarray:
        # _output method needs to be defined for each operation
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # _input_grad method needs to be defined for each operation
        raise NotImplementedError()


'''
A child class for operations that involve parameters (inherits from Operation base class)
'''
class ParamOperation(Operation):

    '''
    constructor
    '''
    def __init__(self, param: np.ndarray):
        super().__init__()
        self.param = param

    '''
    backward pass for operations that take in parameters will also require computing gradients of the output w.r.t. the parameters 
    '''    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:   
        # make sure output gradient arrays and output are the same shape 
        assert_same_shape(self.output, output_grad) 

        # compute gradients w.r.t. inputs and params
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        # make sure gradient arrays are the correct shape 
        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)
        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # _param_grad operator nmeed to be defined for each operator involving parameters
        raise NotImplementedError()



'''
Class for the matrix multiplication operation of inputs with weights
'''
class WeightMultiply(ParamOperation):

    '''
    constructor
    '''
    def __init__(self, W: np.ndarray):
        #initialize with self.param = W
        super().__init__(W)

    '''
    _output method definition
    '''
    def _output(self) -> np.ndarray:
        # comput output
        return np.dot(self.input_, self.param)

    '''
    _input_grad method definition
    '''    
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # comput gradient w.r.t. input (using chain rule)
        return np.dot(output_grad, np.transpose(self.param, (1,0)))

    '''
    _param_grad method definition
    '''
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # compute gradient w.r.t. parameters (using chain rule)
        return np.dot(np.transpose(self.input_, (1,0)), output_grad)


'''
Class for bias addition operator
'''        
class BiasAdd(ParamOperation):
    '''
    constructor
    '''
    def __init__(self, B: np.ndarray):
        # make sure bias array is the correct shape
        assert B.shape[0] == 1, "Incorrect shape for bias array"
        # initialize with self.param = B
        super().__init__(B)

    '''
    _output method definition
    '''
    def _output(self) -> np.ndarray:
        # comput output
        return self.input_ + self.param

    '''
    _input_grad method definition
    '''    
    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # comput gradient w.r.t. input (using chain rule)
        return np.ones_like(self.input_) * output_grad

    '''
    _param_grad method definition
    '''
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # compute gradient w.r.t. parameters (using chain rule), need to sum over rows and transpose
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])

'''
Class for tanh activation function 
'''
class Tanh(Operation):

    '''
    constructor
    '''
    def __init__(self):
        super().__init__()

    '''
    _output method definition
    '''
    def _output(self) -> np.ndarray:
        # comput output
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # compute gradient w.r.t. input (using chain rule)
        return (output_grad * (1.0 - self.output * self.output))


'''
Class for sigmoid activation function 
'''
class Sigmoid(Operation):

    '''
    constructor
    '''
    def __init__(self):
        super().__init__()

    '''
    _output method definition
    '''
    def _output(self) -> np.ndarray:
        # comput output
        return 1.0/(1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # compute gradient w.r.t. input (using chain rule)
        return (self.output * (1.0 - self.output) * output_grad)


'''
Class for linear activation function (output = input) 
'''
class Linear(Operation):

    '''
    constructor
    '''
    def __init__(self):
        super().__init__()

    '''
    _output method definition
    '''
    def _output(self) -> np.ndarray:
        # comput output
        return self.input_

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # compute gradient w.r.t. input (using chain rule)
        return output_grad

'''
Class for a layer of "neurons" in a neural network
'''
class Layer(object):
    '''
    constructor takes in the number of neurons, which is basically just the "breadth" of the layer
    '''
    def __init__(self, neurons: int):
        self.neurons = neurons
        self.first = True
        self.params: List[np.ndarray] = []  # parameters for all the operation
        self.param_grads: List[np.ndarray] = []  # gradients w.r.t. parameters
        self.operations: List[Operation] = []  # all operations within this layer

    def _setup_layer(self, num_in: int) -> None:
        # _setup_layer method needs to be implemented for each layer
        raise NotImplementedError()    

    '''
    forward passes: passes the inputs forward through a series of oprations
    '''    
    def forward(self, input_:np.ndarray) -> np.ndarray:
        
        if self.first:
            self._setup_layer(input_)
            self.first = False

        # initialize layer input
        self.input_ = input_

        # forward passes through each operation
        for operation in self.operations:
            input_ = operation.forward(input_)

        # layer output
        self.output = input_
        return self.output    

    '''
    backward passes: passes output gradient through a series of operations
    '''
    def backward(self, output_grad: np.ndarray) -> np.ndarray: 
 
        # make sure array shapes are correct
        assert_same_shape(self.output, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        self._param_grads()       

        return input_grad

    '''
    method for extracting the parameter gradients from each operation
    '''
    def _param_grads(self) -> None:

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

        return None
    '''
    method for extracting the parameters from each operation
    '''
    def _params(self) -> None:

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)

        return None

'''
A class for a "dense" layer (e.g. Layer 1 of the sigmoid neural network), which is a fully-connected layer, meaning that the output neurons (features) are a combination of all the input neurons (features)
'''
class Dense(Layer):
    '''
    constructor (note that the default activation function is Tanh)
    '''
    def __init__(self, neurons : int, activation: Operation = Tanh()):
        super().__init__(neurons)
        self.activation = activation

    '''
    initialize fully-connected layer attributes
    '''
    def _setup_layer(self, input_: np.ndarray) -> None:
        if (self.seed):
            np.random.seed(self.seed)

        self.params = []
        
        # initialize the weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))
        
        # initialize the bias
        self.params.append(np.random.randn(1, self.neurons))
        
        # initialize the operations
        self.operations = [WeightMultiply(self.params[0]),BiasAdd(self.params[1]), self.activation]

        return None

'''
A base class for the Loss function
'''

class Loss(object): 
    '''
    constructor
    '''
    def __init__(self):
        pass

    '''
    computes the loss function value for given prediction and target
    '''    
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:

        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target

        loss_value = self._output()
        return loss_value

    '''
    computes the gradient of the loss functoin w.r.t. to it's inputs
    '''
    def backward(self) -> np.ndarray:    
        self.input_grad = self._input_grad()
        assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad

    def _output(self):    
        # every subclass of Loss needs to inplement _output method
        raise NotImplementedError()

    def _input_grad(self):   
        # every subclass of Loss needs to implement _input_grad method
        raise NotImplementedError()

'''
A subclass for mean-squred error Loss function
'''
class MeanSquaredError(Loss):
    '''
    constructor
    '''
    def __init__(self):
        super().__init__()

    '''
    compute mean squared loss
    '''
    def _output(self) -> float:    
        return np.sum(np.power(self.prediction - self.target, 2))/self.prediction.shape[0]

    '''
    computes the loss gradient w.r.t. loss input
    '''
    def _input_grad(self) -> np.ndarray:    
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


'''
A subclass for mean-soft-max cross entropy Loss function
'''
class SoftmaxCrossEntropyLoss(Loss):
    '''
    constructor
    '''
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps
        self.single_output = False

    '''
    compute mean squared loss
    '''
    def _output(self) -> float:    
        
        # apply softmax function to each row
        softmax_preds = softmax(self.prediction, sf_axis=1)
        
        # clip softmax output range to (0 + eps, 1 - eps) to prevent numerical overflow
        self.softmax_preds = np.clip(softmax_preds, self.eps, 1 - self.eps)

        # compute the softmax cross entropy loss
        SCE_loss = -1.0 * self.target * np.log(self.softmax_preds) - (1.0 - self.target) * np.log(1 - self.softmax_preds) 
        return np.sum(SCE_loss) / self.prediction.shape[0]
        
    '''
    computes the loss gradient w.r.t. loss input
    '''
    def _input_grad(self) -> np.ndarray:    
        return (self.softmax_preds - self.target) / self.prediction.shape[0]



'''
The Neural Network Class
'''        
class NeuralNetwork(object):
    '''
    construtor (default loss function is MeanSquaredError)
    '''
    def __init__(self, layers: List[Layer], loss: Loss = MeanSquaredError, seed: float = 1):
        # initialize layers and loss function
        self.layers = layers
        self.loss   = loss
        self.seed   = seed

        if(seed):
            for layer in layers:
                setattr(layer, "seed", self.seed)

    '''
    forward passes through a series of layers
    '''      
    def forward(self, X_batch: np.ndarray) -> np.ndarray:
        X_out = X_batch
        for layer in self.layers:
            X_out = layer.forward(X_out)      

        return X_out        

    '''
    backward passes of loss gradients through a series of layers
    '''
    def backward(self, loss_grad: np.ndarray) -> None:
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return None

    '''
    pass inputs forward through the layers, compute loss and pass gradients 
    backward through the layers
    '''
    def train_batch(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
       
        # compute prediction
        prediction = self.forward(X_batch)

        #print("P =")
        #print(prediction)
        
        # compute loss
        loss = self.loss.forward(prediction, y_batch)
        #print(f"Loss = {loss}")

        # compute loss gradients
        self.backward(self.loss.backward())

        return loss

    '''
    extract parameters from the network
    '''
    def params(self):
        for layer in self.layers:   
            yield from layer.params

    '''
    extract loss gradients w.r.t. the parameters from the network
    '''
    def param_grads(self):
        for layer in self.layers:   
            yield from layer.param_grads
 

'''
A base optimizer class for updating the Neural Network parameters based on the loss gradients
'''
class Optimizer(object):
    '''
    constructor (initialize with the learning rate, default = 0.01)
    '''
    def __init__(self, lr : float = 0.01):
        self.lr = lr
        self.first = True

    def step(self):
        # step method needs to be implemented for each optimizer sub class
        raise NotImplementedError()

'''
A Stochastic Gradient Descent Optimizer sub-class
'''
class SGD(Optimizer):
    '''
    constructor
    '''
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    '''
    step method for updating parameters using stochastic gradient decent
    '''    
    def step(self):

        if (self.first):
            self.first = False
        
        # get the params and the corresponding loss gradients and perform updates
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            self._update_rule(param = param, grad = param_grad)

    '''
    update rule for SGD with momentum 
    '''        
    def _update_rule(self, **kwargs):

        update = self.lr * kwargs['grad']
        kwargs['param'] -= update


'''
A Stochastic Gradient Descent Optimizer sub-class with momentum
'''
class SGDMomentum(Optimizer):
    '''
    constructor
    '''
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(lr)
        self.momentum = momentum

    '''
    step method for updating parameters using stochastic gradient decent
    '''    
    def step(self):

        # if this is the first iteration, initialize the "velocities" for each param
        if (self.first):
            self.velocities = [np.ones_like(param) for param in self.net.params()] 
            self.first = False

        # get the params and the corresponding loss gradients and perform updates
        for (param, param_grad, velocity) in zip(self.net.params(), self.net.param_grads(), self.velocities):
            self._update_rule(param = param, grad = param_grad, velocity = velocity)

    '''
    update rule for SGD with momentum 
    '''        
    def _update_rule(self, **kwargs):

        kwargs['velocity'] *= self.momentum
        kwargs['velocity'] += self.lr * kwargs['grad']
        kwargs['param'] -= kwargs['velocity']


'''
A class for training a given neural network using a given optimizer
'''
class Trainer(object):
    '''
    constructor
    '''
    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.net = net
        self.optim = optim
        self.best_loss = 1e9

        # assign the neural networks as an attribute to the optimizer
        setattr(self.optim, 'net', self.net)

    '''
    generator function for creating smaller batches of the training data set
    '''
    Batch = Tuple[np.ndarray, np.ndarray]
    def generate_batch(self, X: np.ndarray, y: np.ndarray, size: int = 32) -> Batch:

        assert X.shape[0] == y.shape[0], "fetaures and target must have same number of rows"        

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch = X[ii:ii+size]    
            y_batch = y[ii:ii+size]    
            yield X_batch, y_batch

    '''
    A 'fit' function for guiding the training process on the given data set. Training steps consist of iteration epochs, in
    each epoch, the data is split into batches, each batch is trained, i.e. passed forward through the layers to compute a prediction, 
    then passed backward to compute loss gradients and then parameters are updated by the optimizer. An epoch ends when all batches have been trained.
    '''
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
            epochs: int = 100, eval_every: int = 10, batch_size: int = 32, seed: int = 1, restart: bool = True):

        np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True

            self.best_loss = 1.0e19

        # iterate over the eopchs
        for e in range(epochs):

            #print(f"Training epoch # {e+1}")
  
            if((e+1)%eval_every == 0):
                # create a copy of the neural network state,
                # will need it for early stopping incase losses start to increase 
                last_model = copy.deepcopy(self.net)

            # permute training data at the beginning of the epoch
            X_train, y_train = permute_data(X_train, y_train)

            # generate batches of the training data
            batch_generator = self.generate_batch(X_train, y_train, batch_size)

            # iterate over batches and train them
            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                #print(f"Training batch # {ii+1}")
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()
            
            # evaluate loss function and errors on the testing data 
            if((e+1)%eval_every == 0):
                test_pred = self.net.forward(X_test)
                test_loss = self.net.loss.forward(test_pred, y_test)

             
                if test_loss< self.best_loss:
                    print(f"Validation loss after {e+1} epochs is {test_loss: .3f}")
                    self.best_loss = test_loss
                    
                else:
                    print(f"""Loss increased after epoch {e+1}, final loss was {self.best_loss:.3f}, using the model from epoch {e+1-eval_every}""")                    
                    
                    # reset the neural network state to what is was at the beginning of the epoch and stop doing iterations
                    self.net = last_model
                    setattr(self.optim, 'net', last_model)
                    break
                

'''
function for evauating error in the prediction
'''
def regression_model_errors(model: NeuralNetwork, X: np.ndarray, y: np.ndarray):

    # compute prediction
    P = model.forward(X)
    
    # compute mean absolute error
    mabs_err = np.mean(np.abs(y-P)) 

    # compute root mean squared error
    rms_err = np.sqrt(np.mean(np.power(y-P, 2)))    

    print(f"Mean absolute error = {mabs_err :e}")
    print(f"Root mean squared error = {rms_err :e}")
