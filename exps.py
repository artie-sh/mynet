import numpy as np
from mnist_loader import load_data
import matplotlib.image as img
import random
import math

training_data, validation_data, test_data = load_data()
training_img = np.asarray(training_data[0])
training_vals = np.asarray(training_data[1])

def save_as_png(dataset, number):
    folder = './images/'
    for i in range(number):
        img.imsave(folder + '%s__%s.png' % (str(i), str(training_vals[i])), training_img[i].reshape(28,28))


def init_weights(number):
    return [0 for i in range(number)]

def init_b():
    return random.random()

def calc_z(W, X, b):
    sum = 0
    assert len(W) == len(X)
    for i in range(len(W)):
        sum += W[i] * X[i]
    return sum + b

def calc_sigmoid(z):
    return 1/(1+math.exp(-z))

def calc_cost(Y, A):
    sum = 0
    assert len(Y) == len(A)
    for i in range(len(A)):
       sum += Y[i] * math.log(A[i]) + (1 - Y[i]) * math.log(1 - A[i])
    return -sum/len(A)

def calc_loss(Y, A):
    - Y * math.log(A) - (1 - Y) * math.log(1 - A)

def normalize_val(val):
    result = 0
    if val == 4:
        result = 1
    return result


def calc_dw(X, A, Y):
    dws = []
    A_Y = [A[i] - Y[i] for i in range(len(A))]

    for i in range(len(X[0])):
        sum = 0
        for j in range(len(A_Y)):
            sum += X[j][i] * A_Y[j]
        dws.append(sum/len(X))
    return dws

def calc_db(Y, A):
    sum = 0
    for i in range(len(Y)):
        sum += A[i] - Y[i]
    return sum/len(Y)


def propagate(W, b, X, Y):
    A = [calc_sigmoid(calc_z(W, X[i], b)) for i in range(rng)]
    cost = calc_cost(Y, A)
    # grads = {'dw': calc_dw(X, A, Y), 'db': calc_db(Y, A)}
    dw = calc_dw(X, A, Y)
    db = calc_db(Y, A)
    return dw, db, cost


def optimize(W, b, X, Y):
    num_iterations = 100
    learning_rate = 0.1

    for i in range(num_iterations):
        dw, db, cost = propagate(W, b, X, Y)
        for j in range(len(W)):
            W[j] -= learning_rate * dw[j]
        db -= learning_rate * db
    return cost

rng = 100

W = init_weights(784)
b = 0
X = [training_img[i] for i in range(rng)]
Y = [normalize_val(training_vals[i]) for i in range(rng)]

print optimize(W, b, X, Y)

