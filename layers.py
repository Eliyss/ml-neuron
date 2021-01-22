import numpy as np 
import functions

class InputLayer:
    def __init__(self, num_input):
        self.num_input = num_input
        
    def predict(self, x):
        return x

class Layer:
    def __init__(self, num_input, num_neuron, func='reLu'):
        self.func = func
        self.weights = np.random.rand(num_neuron, num_input)
        self.bias = np.random.rand(num_neuron, 1)
    
    def update(self, update_grad, update_bias, lr):
        grad_norm = np.linalg.norm(update_grad, ord=np.inf)
        bias_norm = np.linalg.norm(update_bias, ord=np.inf)
        if grad_norm > 2:
            update_grad = update_grad/grad_norm*2
                                   
        if bias_norm > 2:
            update_grad = update_bias/bias_norm*2                           
                                   
        self.weights = np.add(self.weights, lr*update_grad)
        self.bias = np.add(self.bias, lr*update_bias)
    
    def backError(self):
        if self.func == 'reLu':
            return np.vectorize(functions.dReLu)(self.z)
        elif self.func == 'sigmoid':
            return np.vectorize(functions.dsigmoid)(self.z)
        elif self.func == 'softmax':
            return functions.dsoftmax(self.z)
        
        
    def predict(self, prev_activation):
        self.z = np.matmul(self.weights, prev_activation)
        if self.func == 'reLu':
            return np.vectorize(functions.reLu)(np.add(self.z, self.bias))
        elif self.func == 'sigmoid':
            return np.vectorize(functions.sigmoid)(np.add(self.z, self.bias))
        elif self.func == 'softmax':
            return functions.softmax(np.add(self.z, self.bias))