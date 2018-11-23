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


def init_weights(number):
    return [0 for i in range(number)]

def init_b():
    return 0


def calc_z(W, X, b):
    rng = len(W)
    sum = 0
    assert rng == len(X)
    for i in range(rng):
        sum += W[i] * X[i]
    return sum + b


def calc_sigmoid(z):
    return 1/(1+math.exp(-z))


def calc_cost(Y, A):
    rng = len(A)
    sum = 0
    assert len(Y) == rng
    for i in range(rng):
       sum += Y[i] * math.log(A[i]) + (1 - Y[i]) * math.log(1 - A[i])
    return -sum/rng



def normalize_val(val, target_val):
    result = 0
    if val == target_val:
        result = 1
    return result


def calc_dw(X, A, Y):
    rng = len(A)
    lng = len(X[0])
    dws = []
    A_Y = [A[i] - Y[i] for i in range(rng)]

    for i in range(lng):
        sum = 0
        for j in range(rng):
            sum += X[j][i] * A_Y[j]
        dws.append(sum/lng)
    return dws


def calc_db(Y, A):
    rng = len(Y)
    sum = 0
    for i in range(rng):
        sum += A[i] - Y[i]
    return sum/rng


def propagate(W, b, X, Y):
    A = [calc_sigmoid(calc_z(W, X[i], b)) for i in range(len(X))]
    cost = calc_cost(Y, A)
    dw = calc_dw(X, A, Y)
    db = calc_db(Y, A)
    return dw, db, cost


def optimize(W, b, X, Y, learning_rate, num_iterations):
    for i in range(num_iterations):
        dw, db, cost = propagate(W, b, X, Y)
        for j in range(len(W)):
            W[j] -= learning_rate * dw[j]
        b -= learning_rate * db
        if i % 10 == 0:
            print "%s: cost %s" % (str(i), str(cost))
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
X = [training_img[i] for i in range(trainig_sets)]
Y = [normalize_val(training_vals[i], target_number) for i in range(trainig_sets)]

W, b = optimize(W, b, X, Y, learning_rate, num_iterations)

true_rec, false_rec, true_unrec, false_unrec = 0, 0, 0, 0

for i in range(400, 500):
    result = str(predict(W, b, training_img[i]))
    fact = str(training_vals[i])

    if result >= 0.5 and fact == target_number:
        print "%s recognized %s - %s" % (str(i), result, fact)
        true_rec += 1
    elif result >= 0.5 and fact != target_number:
        print "%s false rec %s - %s" % (str(i), result, fact)
        false_rec += 1
    elif result < 0.5 and fact != target_number:
        print "%s true unrec %s - %s" % (str(i), result, fact)
        true_unrec += 1
    elif result < 0.5 and fact == target_number:
        print "%s unrecognized %s - %s" % (str(i), result, fact)
        false_unrec += 1


print "true_rec", true_rec
print "false_rec", false_rec
print "true_unrec", true_unrec
print "false_unrec", false_unrec


track_end(start)
