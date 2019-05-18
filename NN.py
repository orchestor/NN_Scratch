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
