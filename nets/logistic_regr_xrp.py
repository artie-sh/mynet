from database_writer import CsvReader
from params import *
from my_network import VectorizedNet

class XrpDataHandler():

    data, x_axis, y_axis, ma, normalized_ma, minimum, maximum  = None, None, None, None, None, None, None

    def __init__(self, tablename, column):
        csv = CsvReader()
        conn = csv.connect_db(csv.db_path)
        cur = conn.cursor()
        self.data = csv.read_db(cur, tablename)
        self.x_axis, self.y_axis = self.get_timerow(column)

    def avg(self, timerow):
        return sum(timerow) / max(len(timerow), 1)

    def get_timerow(self, rowname='close'):
        rows = {'open': 1, 'high': 2, 'low': 3, 'close': 4}
        colnum = rows[rowname]
        return np.array([line[0] for line in self.data]), np.array([line[colnum] for line in self.data])

    def get_ma(self, periods):
        result = np.concatenate((np.array([None] * (periods - 1)), np.array([self.avg(self.y_axis[i:i + periods]) for i in range(len(self.y_axis) - periods)])))
        return result

    def normalize_vals(self, timerow):
        self.minimum, self.maximum = min(timerow), max(timerow)
        return (timerow - self.minimum) / (self.maximum - self.minimum)

    def get_X_train(self, timerow, input_size, training_sets):
        return np.array([timerow[i:i+input_size] for i in range(training_sets)])

    def get_Y_train(self, timerow, input_size, training_sets):
        return np.array([timerow[i + input_size] for i in range(training_sets)])

    def get_last_X_for_pred(self, input_size):
        vals = np.array(self.normalized_ma[len(xrp.normalized_ma) - input_size:])
        return vals.reshape(input_size, 1)

    def unwrap_prediction(self, prediction):
        return (prediction * (self.maximum - self.minimum) + self.minimum)[0]



# start = track_start()
#
# xrp = XrpDataHandler('xrp', 'close')
# net = VectorizedNet(input_size=50, trainig_sets=10000, num_iterations=2000, learning_rate=0.3)
#
# xrp.normalized_ma = xrp.normalize_vals(xrp.get_ma(4))
#
# X = xrp.get_X_train(xrp.normalized_ma, net.input_size, net.trainig_sets)
# Y = xrp.get_Y_train(xrp.normalized_ma, net.input_size, net.trainig_sets)
#
# net.optimize(X, Y)
#
# print xrp.unwrap_prediction(net.predict_monosign(xrp.get_last_X_for_pred(net.input_size)))
#
# track_end(start)