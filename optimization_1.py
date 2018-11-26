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
    return np.zeros((number, 1))

def init_b():
    return 0


def calc_z(W, X, b):
    res = np.dot(W.T, X.reshape(784, len(X))) + b
    return res


def calc_sigmoid(z):
    sigmoid = 1/(1+np.exp(-z))
    return sigmoid


def calc_cost(Y, A):
    A = A.T
    sum = np.dot(Y, np.log(A)) + np.dot((1 - Y), np.log(1 - A))
    return -sum[0]/len(Y)



def normalize_val(val, target_val):
    result = 0
    if val == target_val:
        result = 1
    return result


def calc_dw(X, A, Y):
    result = np.dot((A - Y), X)
    print result.shape
    return result


def calc_db(Y, A):
    return np.sum(A - Y) / len(Y)


def propagate(W, b, X, Y):
    z = calc_z(W, X, b)
    A = calc_sigmoid(z)
    cost = calc_cost(Y, A)
    dw = calc_dw(X, A, Y)
    db = calc_db(Y, A)
    return dw, db, cost


def optimize(W, b, X, Y, learning_rate, num_iterations):
    for i in range(num_iterations):
        dw, db, cost = propagate(W, b, X, Y)
        W -= np.dot(dw.T, learning_rate)
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
X = np.array([training_img[i] for i in range(trainig_sets)])
Y = np.array([normalize_val(training_vals[i], target_number) for i in range(trainig_sets)])

W, b = optimize(W, b, X, Y, learning_rate, num_iterations)


true_rec, false_rec, true_unrec, false_unrec = 0, 0, 0, 0

result = predict(W, b, np.array([training_img[i] for i in range(400, 500)]))[0]
fact = [training_vals[i] for i in range(400, 500)]

# for i in range(len(result)):
#     print "result: %s, fact: %s" % (str(result[i]), str(fact[i]))

#     if result >= 0.5 and fact == target_number:
#         print "%s recognized %s - %s" % (str(i), result, fact)
#         true_rec += 1
#     elif result >= 0.5 and fact != target_number:
#         print "%s false rec %s - %s" % (str(i), result, fact)
#         false_rec += 1
#     elif result < 0.5 and fact != target_number:
#         print "%s true unrec %s - %s" % (str(i), result, fact)
#         true_unrec += 1
#     elif result < 0.5 and fact == target_number:
#         print "%s unrecognized %s - %s" % (str(i), result, fact)
#         false_unrec += 1
#
#
# print "true_rec", true_rec
# print "false_rec", false_rec
# print "true_unrec", true_unrec
# print "false_unrec", false_unrec


track_end(start)
