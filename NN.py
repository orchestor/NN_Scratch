import numpy as np


class Sigmod:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


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
