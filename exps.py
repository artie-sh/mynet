import numpy as np

def calc_z(W, X, b):
    return np.dot(W, X) + b


W = np.array([1, 2, 3, 4])
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
b = 1

print calc_z(W, X, b)