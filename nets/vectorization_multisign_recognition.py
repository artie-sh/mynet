from params import *
from my_network import VectorizedNet


start = track_start()

net = VectorizedNet()
net.load_data()

W = np.array([net.init_weights(784) for i in range(10)])
b = [0 for i in range(10)]
X = np.array([net.training_img[i] for i in range(trainig_sets)])
Y = np.array([[net.normalize_val(net.training_vals[i], j) for i in range(trainig_sets)] for j in range(10)])

for i in range(10):
    print "running optimization on %s" % str(i)
    W[i], b[i] = net.optimize(W[i], b[i], X, Y[i], learning_rate, num_iterations)

result = net.predict_multisign(W, [net.training_img[i] for i in range(10000, 10100)], b)
fact = [net.training_vals[i] for i in range(10000, 10100)]

rec, unrec = 0, 0

for i in range(len(result)):
    if result[i][0] == fact[i]:
        rec += 1
        append = ''
    else:
        unrec += 1
        append = '!!!'
    print "res: %s (%s), fact: %s %s" % (str(result[i][0]), str(result[i][1]), str(fact[i]), append)


print "correct: %s" % str(rec)
print "incorrect: %s" % str(unrec)

for i in range(len(W)):
    net.save_idealview(i, W[i])

track_end(start)
