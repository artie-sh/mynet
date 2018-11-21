import numpy as np
from mnist_loader import load_data, load_data_wrapper
import matplotlib.pyplot as plt
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
    return [random.random() for i in range(number)]

def init_b():
    return random.random()

def calc_z(w, x, b):
    sum = []
    assert len(w) == len(x)
    for i in range(len(w)):
        #print '%s * %s + %s = %s' % (str(w[i]), str(x[i]), str(b), str(w[i]*x[i]+b))
        sum.append(w[i]*x[i] + b)
        #print sum
    return sum

def calc_sigmoid(z):
    return [1/(1+math.exp(-i)) for i in z]

def calc_cost(y, a):
    sum = 0
    assert len(y) == len(a)
    for i in range(len(y)):
        sum += y[i]*math.log(a[i]) + (1-y[i])*math.log(1-a[i])
    return -sum/len(y)


w = init_weights(784)

b = init_b()
x = training_img[2]

predictions = calc_sigmoid(calc_z(w, x, b))
cost = calc_cost()
