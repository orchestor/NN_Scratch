import numpy as np

input = np.random.randn(10, 2)    # 10 samples and each sample has two features
w1 = np.random.randn(2,4)         # 4 judges
bias = np.random.randn(4)
output = np.dot(input,w1) + bias
print(output)
