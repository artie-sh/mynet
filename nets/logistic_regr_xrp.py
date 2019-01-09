from database_writer import CsvReader
from params import *
from my_network import VectorizedNet

class LogisticRegressionXRP(VectorizedNet):

    def avg(self, timerow):
        return sum(timerow) / max(len(timerow), 1)

    def get_timerow(self, data, rowname='close'):
        rows = {'open': 1, 'high': 2, 'low': 3, 'close': 4}
        colnum = rows[rowname]
        return [line[colnum] for line in data]

    def get_ma(self, timerow, periods):
        return np.array([self.avg(timerow[i:i + periods]) for i in range(len(timerow) - periods)])

    def get_normalized_ma(self, timerow, periods):
        ma = self.get_ma(timerow, periods)
        return np.array([normalized_ma[i:i+input_size] for i in range(trainig_sets)])

    def normalize_vals(self, timerow, min, max):
        return (timerow - min) / (max - min)

    def unwrap_prediction(self, prediction, minimum, maximum):
        return prediction * (maximum - minimum) + minimum


start = track_start()

csv = CsvReader()
conn = csv.connect_db(csv.db_path)
cur = conn.cursor()
data = csv.read_db(cur, 'xrp')
net = LogisticRegressionXRP()

input_size = 50
trainig_sets = 10000
num_iterations = 2000
learning_rate = 0.3

timerow = net.get_timerow(data, 'close')
ma = net.get_ma(timerow, 3)
minimum, maximum = min(ma), max(ma)
normalized_ma = net.normalize_vals(ma, minimum, maximum)

W = net.init_weights(input_size)
b = 0

X = np.array([normalized_ma[i:i+input_size] for i in range(trainig_sets)])
Y = np.array([normalized_ma[i+input_size] for i in range(trainig_sets)])

W, b = net.optimize(W, b, X, Y, learning_rate, num_iterations)

vals = np.array(normalized_ma[len(normalized_ma)-input_size:])
vals = vals.reshape(input_size, 1)

prediction = net.predict_monosign(W, vals.T, b)


print net.unwrap_prediction(prediction, minimum, maximum)


track_end(start)