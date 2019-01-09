from params import *
from my_network import NonVectorizedNet


start = track_start()

net = NonVectorizedNet()
net.load_data()

W = [net.init_weights(784) for i in range(10)]
b = [0 for i in range(10)]
X = [net.training_img[i] for i in range(trainig_sets)]
Y = [[net.normalize_val(net.training_vals[i], j) for i in range(trainig_sets)] for j in range(10)]

for i in range(len(Y)):
    W[i], b[i] = net.optimize(W[i], b[i], X, Y[i], learning_rate, num_iterations)

correct, wrong = 0, 0

for i in range(400, 500):
    result, prob = net.predict(W, b, net.training_img[i])
    fact = net.training_vals[i]
    print 'result: %s, fact: %s' % (result, fact)
    if result == fact:
        correct += 1
    else:
        wrong += 1

print 'correct ', correct
print 'wrong ', wrong

track_end(start)
