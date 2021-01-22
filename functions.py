import numpy as np 

def reLu(x):
    return np.maximum(0, x)
    
def dReLu(x):
    return 1*(x>0)
    
def sigmoid(x):
    return 1/(1+np.exp(-x)) 

def dsigmoid(x):
    f = 1/(1+np.exp(-x)) 
    return f*(1-f)

def softmax(x):
    exps = np.exp(x-np.max(x))
    return exps/exps.sum()

def dsoftmax(x):
    s = x.reshape(-1,1)
    deriv = np.diagflat(s)-np.dot(s, s.T)
    return deriv.diagonal().reshape(-1,1)
