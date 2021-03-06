from  params import *


training_data, validation_data, test_data = load_data()
training_img = np.asarray(training_data[0])
training_vals = np.asarray(training_data[1])

def save_as_png(dataset, number):
    folder = './images/'
    for i in range(number):
        img.imsave(folder + '%s__%s.png' % (str(i), str(training_vals[i])), training_img[i].reshape(28,28))


def save_idealview(number, W):
    for i in range(len(W)):
        if W[i] < 0.01:
            W[i] = 0
    W = np.abs(W-1)
    folder = './idealview/'
    img.imsave(folder + '%s.png' % str(number), W.reshape(28,28), cmap=cm.gray)


def init_weights(number):
    return np.array([0. for i in range(number)])


def init_b():
    return 0

def normalize_val(val, target_val):
    result = 0
    if val == target_val:
        result = 1
    return result


def calc_z(W, X, b):
    return np.sum(W * X, 1) + b


def calc_sigmoid(z):
    return 1/(1+np.exp(-z))


def calc_cost(Y, A):
    return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))/A.shape[0]


def calc_dw(X, A, Y):
    return np.dot(A - Y, X) / len(A)


def calc_db(Y, A):
    return np.sum(A - Y) / A.shape[0]


def propagate(W, b, X, Y):
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
        # if i % 10 == 0:
        #     print "%s: cost %s" % (str(i), str(cost))
    print "done, final cost %s" % str(cost)
    return W, b


def predict(W, X, b):
    sigmoids = np.array([calc_sigmoid(calc_z(W[i], X, b[i])) for i in range(len(W))])
    return [(sigmoids.argmax(0)[i], sigmoids.max(0)[i]) for i in range(sigmoids.shape[1])]


start = track_start()

W = np.array([init_weights(784) for i in range(10)])
b = [0 for i in range(10)]
X = np.array([training_img[i] for i in range(trainig_sets)])
Y = np.array([[normalize_val(training_vals[i], j) for i in range(trainig_sets)] for j in range(10)])

for i in range(10):
    print "running optimization on %s" % str(i)
    W[i], b[i] = optimize(W[i], b[i], X, Y[i], learning_rate, num_iterations)

# result = predict(W, [training_img[i] for i in range(10000, 10100)], b)
# fact = [training_vals[i] for i in range(10000, 10100)]
#
# rec, unrec = 0, 0
#
# for i in range(len(result)):
#     if result[i][0] == fact[i]:
#         rec += 1
#         append = ''
#     else:
#         unrec += 1
#         append = '!!!'
#     print "res: %s (%s), fact: %s %s" % (str(result[i][0]), str(result[i][1]), str(fact[i]), append)
#
#
# print "correct: %s" % str(rec)
# print "incorrect: %s" % str(unrec)

for i in range(len(W)):
    save_idealview(i, W[i])

track_end(start)
