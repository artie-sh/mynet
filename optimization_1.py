import numpy as np
from mnist_loader import load_data
import matplotlib.image as img
import random
import math
from datetime import datetime as dt
from datetime import timedelta
import params


training_data, validation_data, test_data = load_data()
training_img = np.asarray(training_data[0])
training_vals = np.asarray(training_data[1])

def save_as_png(dataset, number):
    folder = './images/'
    for i in range(number):
        img.imsave(folder + '%s__%s.png' % (str(i), str(training_vals[i])), training_img[i].reshape(28,28))

#save_as_png(training_img, 1000)


def init_weights(number):
    return np.array([0. for i in range(number)])


def init_b():
    return 0

def calc_z(W, X, b):
    res = np.sum(W * X, 1) + b
    return res


def calc_sigmoid(z):
    res = 1/(1+np.exp(-z))
    return res


def calc_cost(Y, A):
    res = np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    return -res/A.shape[0]


def normalize_val(val, target_val):
    result = 0
    if val == target_val:
        result = 1
    return result


def calc_dw(X, A, Y):
    X = X.reshape(784, A.shape[0])
    return np.sum((A-Y) * X, 1)/A.shape[0]


def calc_db(Y, A):
    res = np.sum(A - Y) / A.shape[0]
    return res


def propagate(W, b, X, Y):
    #A = [calc_sigmoid(calc_z(W, X[i], b)) for i in range(len(X))]
    A = calc_sigmoid(calc_z(W, X, b))
    cost = calc_cost(Y, A)
    dw = calc_dw(X, A, Y)
    db = calc_db(Y, A)
    return dw, db, cost


def optimize(W, b, X, Y, learning_rate, num_iterations):
    for i in range(num_iterations):
        dw, db, cost = propagate(W, b, X, Y)
        W -= dw * learning_rate
        b -= db * learning_rate
        if i % 10 == 0:
            print "%s: cost %s" % (str(i), str(cost))
            #print "db st", db
    return W, b


def predict(W, b, x):
    return calc_sigmoid(calc_z(W, x, b))


def track_start():
    start = dt.now()
    print "start time: %s" % str(start.strftime('%Y-%m-%d %H:%M:%S'))
    return start


def track_end(start):
    end = dt.now()
    delta = end - start
    print "end time is %s" % end.strftime('%Y-%m-%d %H:%M:%S')
    print "total duration: %s" % (str(delta-timedelta(microseconds=delta.microseconds)))




start = track_start()

target_number = params.target_number
trainig_sets = params.trainig_sets
num_iterations = params.num_iterations
learning_rate = params.learning_rate

W = init_weights(784)
b = 0
X = np.array([training_img[i] for i in range(trainig_sets)])
Y = np.array([normalize_val(training_vals[i], target_number) for i in range(trainig_sets)])

W, b = optimize(W, b, X, Y, learning_rate, num_iterations)

true_rec, false_rec, true_unrec, false_unrec = 0, 0, 0, 0

result = predict(W, b, [training_img[i] for i in range(400, 500)])
fact = [training_vals[i] for i in range(400, 500)]

for i in range(len(result)):
    if result[i] >= 0.5 and fact[i] == target_number:
        print "%s recognized %s - %s" % (str(i), str(result), str(fact))
        true_rec += 1
    elif result[i] >= 0.5 and fact[i] != target_number:
        print "%s false rec %s - %s" % (str(i), str(result), str(fact))
        false_rec += 1
    elif result[i] < 0.5 and fact[i] != target_number:
        print "%s true unrec %s - %s" % (str(i), str(result), str(fact))
        true_unrec += 1
    elif result[i] < 0.5 and fact[i] == target_number:
        print "%s unrecognized %s - %s" % (str(i), str(result), str(fact))
        false_unrec += 1


print "true_rec", true_rec
print "false_rec", false_rec
print "true_unrec", true_unrec
print "false_unrec", false_unrec


track_end(start)
