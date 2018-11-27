import numpy as np

def calc_z_old(W, X, b):
    sum = 0
    assert len(W) == len(X)
    for i in range(len(W)):
        sum += W[i] * X[i]
    return sum + b


def calc_z(W, X, b):
    sum = W * X
    return np.sum(W * X, 1) + b


w = np.array([0.1, 0.2, 0.3])
x = np.array([[10, 20, 30], [100, 200, 300]])

b = 0

print w
print w.shape

print x
print x.shape

print calc_z(w, x, b)
#print calc_z_old(w, x, b)

# [0.1 0.2 0.3]
# [[ 10  20  30]
#  [100 200 300]]
# [[ 1.  4.  9.]
#  [10. 40. 90.]]