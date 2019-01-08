from params import *


class SimpleNet():
    training_data, validation_data, test_data, training_img, training_vals = None, None, None, None, None

    def __init__(self):
        self.training_data, self.validation_data, self.est_data = load_data()
        self.training_img = np.asarray(self.training_data[0])
        self.training_vals = np.asarray(self.training_data[1])

    def save_as_png(self, dataset, number):
        folder = '../images/'
        for i in range(number):
            img.imsave(folder + '%s__%s.png' % (str(i), str(dataset[i])), dataset[i].reshape(28, 28))


    def init_weights(self, number):
        return [0 for i in range(number)]


    def init_b(self):
        return 0


    def calc_z(self, W, X, b):
        sum = 0
        assert len(W) == len(X)
        for i in range(len(W)):
            sum += W[i] * X[i]
        return sum + b


    def calc_sigmoid(self, z):
        return 1 / (1 + math.exp(-z))


    def calc_cost(self, Y, A):
        sum = 0
        assert len(Y) == len(A)
        for i in range(len(A)):
            sum += Y[i] * math.log(A[i]) + (1 - Y[i]) * math.log(1 - A[i])
        return -sum / len(A)


    def normalize_val(self, val, target_val):
        result = 0
        if val == target_val:
            result = 1
        return result


    def calc_dw(self, X, A, Y):
        dws = []
        A_Y = [A[i] - Y[i] for i in range(len(A))]

        for i in range(len(X[0])):
            sum = 0
            for j in range(len(A_Y)):
                sum += X[j][i] * A_Y[j]
            dws.append(sum / len(X))
        return dws


    def calc_db(self, Y, A):
        sum = 0
        for i in range(len(Y)):
            sum += A[i] - Y[i]
        return sum / len(Y)


    def propagate(self, W, b, X, Y):
        A = [self.calc_sigmoid(self.calc_z(W, X[i], b)) for i in range(len(X))]
        cost = self.calc_cost(Y, A)
        dw = self.calc_dw(X, A, Y)
        db = self.calc_db(Y, A)
        return dw, db, cost


    def optimize(self, W, b, X, Y, learning_rate, num_iterations):
        for i in range(num_iterations):
            dw, db, cost = self.propagate(W, b, X, Y)
            for j in range(len(W)):
                W[j] -= learning_rate * dw[j]
            b -= learning_rate * db
            if i % 10 == 0:
                print "%s: cost %s" % (str(i), str(cost))
                # print "db st", db
        return W, b


    def predict(self, W, b, x):
        return self.calc_sigmoid(self.calc_z(W, x, b))