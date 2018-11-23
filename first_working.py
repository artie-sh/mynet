import numpy as np
from mnist_loader import load_data
import matplotlib.image as img
import random
import math
from datetime import datetime as dt
from datetime import timedelta

training_data, validation_data, test_data = load_data()
training_img = np.asarray(training_data[0])
training_vals = np.asarray(training_data[1])

def save_as_png(dataset, number):
    folder = './images/'
    for i in range(number):
        img.imsave(folder + '%s__%s.png' % (str(i), str(training_vals[i])), training_img[i].reshape(28,28))

#save_as_png(training_img, 1000)


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


# def calc_loss(Y, A):
#     - Y * math.log(A) - (1 - Y) * math.log(1 - A)


def normalize_val(val, target_val):
    result = 0
    if val == target_val:
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

target_number = 5
trainig_sets = 300
num_iterations = 100
learning_rate = 0.1

W = init_weights(784)
b = 0
X = [training_img[i] for i in range(trainig_sets)]
Y = [normalize_val(training_vals[i], target_number) for i in range(trainig_sets)]

W, b = optimize(W, b, X, Y, learning_rate, num_iterations)

true_rec, false_rec, true_unrec, false_unrec = 0, 0, 0, 0

for i in range(400, 500):
    result = predict(W, b, training_img[i])
    fact = training_vals[i]

    if result >= 0.5 and fact == target_number:
        print "%s recognized %s - %s" % (str(i), str(result), str(fact))
        true_rec += 1
    elif result >= 0.5 and fact != target_number:
        print "%s false rec %s - %s" % (str(i), str(result), str(fact))
        false_rec += 1
    elif result < 0.5 and fact != target_number:
        print "%s true unrec %s - %s" % (str(i), str(result), str(fact))
        true_unrec += 1
    elif result < 0.5 and fact == target_number:
        print "%s unrecognized %s - %s" % (str(i), str(result), str(fact))
        false_unrec += 1


print "true_rec", true_rec
print "false_rec", false_rec
print "true_unrec", true_unrec
print "false_unrec", false_unrec


track_end(start)
