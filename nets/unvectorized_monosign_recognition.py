from params import *
from my_network import NonVectorizedNet

start = track_start()

net = NonVectorizedNet()
net.load_data()

W = net.init_weights(784)
b = net.init_b()
X = [net.training_img[i] for i in range(trainig_sets)]
Y = [net.normalize_val(net.training_vals[i], target_number) for i in range(trainig_sets)]

W, b = net.optimize(W, b, X, Y, learning_rate, num_iterations)

print b

true_rec, false_rec, true_unrec, false_unrec = 0, 0, 0, 0

for i in range(400, 500):
    result = net.predict(W, b, net.training_img[i])
    fact = net.training_vals[i]

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
