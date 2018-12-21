from  params import *



def init_hidden_layer(inputs_num, neurons_num):
    return np.random.random((inputs_num, neurons_num)) * 0.01, np.zeros(neurons_num)

def init_outer_layer(inputs_num):
    return np.zeros((inputs_num, 1)), 0

def calc_zs(A, W, b):
    assert A.shape[1] == W.shape[0]
    #print A.shape
    #print W.shape
    return np.dot(A, W) + b

def calc_relus(Z):
    return np.maximum(Z, 0)

def calc_sigmoid(z):
    return 1/(1+np.exp(-z))


def calc_cost(Y, A):
    return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))/A.shape[0]

def normalize_val(val, target_val):
    result = 0
    if val == target_val:
        result = 1
    return result


training_data, validation_data, test_data = load_data()
training_img = np.asarray(training_data[0])
training_vals = np.asarray(training_data[1])


start = track_start()

X = np.array([training_img[i] for i in range(trainig_sets)])
Y = np.array([normalize_val(training_vals[i], target_number) for i in range(trainig_sets)])

W1, b1 = init_hidden_layer(784, 3)
W2, b2 = init_outer_layer(3)

relus = calc_relus(calc_zs(X, W1, b1))
#print relus

A = calc_sigmoid(calc_zs(relus, W2, b2))
print calc_cost(Y, A)
#print sigmoid






track_end(start)
