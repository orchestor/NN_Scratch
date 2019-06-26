import numpy as np

def sigmod(x):
    return 1 / (1+np.exp(x))

input = np.random.randn(10, 2)    # 10 samples and each sample has two features
w1 = np.random.randn(2,4)         # 4 judges
b1 = np.random.randn(4)
h = np.dot(input,w1) + b1

w2 = np.random.randn(4,3)
b2 = np.random.randn(3)

a = sigmod(h)
output = np.dot(a, w2) + b2
print(output)
