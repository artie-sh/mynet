from params import *
from my_network import VectorizedNet


start = track_start()

net = VectorizedNet(784, trainig_sets, num_iterations, learning_rate)
net.load_data()

X = np.array([net.training_img[i] for i in range(trainig_sets)])
Y = np.array([net.normalize_val(net.training_vals[i], target_number) for i in range(trainig_sets)])

W, b = net.optimize(X, Y)

true_rec, false_rec, true_unrec, false_unrec = 0, 0, 0, 0

result = net.predict_monosign([net.training_img[i] for i in range(400, 500)])
fact = [net.training_vals[i] for i in range(400, 500)]

for i in range(len(result)):
    if result[i] >= 0.5 and fact[i] == target_number:
        print "%s recognized %s - %s" % (str(i), str(result[i]), str(fact[i]))
        true_rec += 1
    elif result[i] >= 0.5 and fact[i] != target_number:
        print "%s false rec %s - %s" % (str(i), str(result[i]), str(fact[i]))
        false_rec += 1
    elif result[i] < 0.5 and fact[i] != target_number:
        print "%s true unrec %s - %s" % (str(i), str(result[i]), str(fact[i]))
        true_unrec += 1
    elif result[i] < 0.5 and fact[i] == target_number:
        print "%s unrecognized %s - %s" % (str(i), str(result[i]), str(fact[i]))
        false_unrec += 1

print "true_rec", true_rec
print "false_rec", false_rec
print "true_unrec", true_unrec
print "false_unrec", false_unrec

net.save_idealview(target_number, W)

track_end(start)
