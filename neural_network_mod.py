'''
    A self-contained multi-layer neural network module
'''

import numpy as np
from typing import Callable, Dict, Tuple, List

'''
function that checks if two arrays have atching shape
'''
def assert_same_shape(array1: np.ndarray, array2: np.ndarray) -> None:
    assert array1.shape == array2.shape, "Array shapes do not match"
    return None

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
    backward pass for operations that take in parameters will also require 
    computing gradients of the output w.r.t. the parameters 
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
        return np.dot(output_grad, np.transpose(self.param,(1,0)))

    '''
    _param_grad method definition
    '''
    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # compute gradient w.r.t. parameters (using chain rule)
        return np.dot(np.transpose(self._input,(1,0)), output_grad)


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
        return 1.0/(1.0 + np.exp(-1.0*self.input_))

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        # compute gradient w.r.t. input (using chain rule)
        return self.output * (1.0 - self.output) * output_grad


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
    def _param_grads(self) -> None:

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)

        return None

'''
A class for a dense layer (e.g. Layer 1 of the sigmoid neural network), a fully-connected layer
in which the output neurons (features) are a combination of all the input neurons (features)
'''
class Dense(Layer):
    '''
    constructor
    '''
    def __init__(self, neurons : int, activation: Operation = Sigmoid()) -> None:
        super().__init__(neurons)
        self.activation = activation

    '''
    initialize layer attributes
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
        return np.mean(np.power(self.prediction - self.target, 2))

    '''
    computes the loss gradient w.r.t. loss input
    '''
    def _input_grad(self) -> np.ndarray:    
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]

'''
The Neural Network Class
'''        
class NeuralNetwork(object):
    '''
    construtor
    '''
    def __init__(self, layers: List[Layer], loss: Loss, seed: float = 1):
        # initialize layers and loss function
        self.layers = layers
        self.loss = loss
        self.seed = seed

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
        # compute prediciton
        prediction = self.forward(X_batch)
        
        # compute loss
        loss = self.loss.forward(prediction, y_batch)

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


