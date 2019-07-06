import numpy as np


class FullyConnectLayer:
    def __init__(self, weight, bias):
        self.params = [weight, bias]

    def forward(self, x):
        weight, bias = self.params
        return np.dot(x, weight) + bias

class Relu:
    def __init_(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        forward_val = x.copy()
        forward_val(self.mask) = 0 # if x is <= zero, then it will be zero. 
        return forward_val

    def backward(self, dout):
        dout(self.mask) = 0
        backward_val = dout
        return backward_val

class Sigmoid:
    def __init_(self):
        self.y = None

    def forward(self, x):
        y = 1/(1+np.exp(-x))
        self.y = y
        return y
    def backward(self,dy):
        dx = dy*(1-self.out)*self.out
        return dx
