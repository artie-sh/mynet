from database_writer import CsvReader
from vectorization_working import *

def avg(timerow):
    return sum(timerow)/max(len(timerow), 1)

def get_timerow(data, rowname='close'):
    rows = {'open': 1, 'high': 2, 'low': 3, 'close': 4}
    colnum = rows[rowname]
    return [line[colnum] for line in data]

def get_ma(timerow, periods):
    return np.array([avg(timerow[i:i + periods]) for i in range(len(timerow) - periods)])

def normalize_vals(timerow, min, max):
    return (timerow - min) / (max - min)

def unwrap_prediction(prediction, minimum, maximum):
    return prediction * (maximum - minimum) + minimum


start = track_start()

csv = CsvReader()
conn = csv.connect_db(csv.db_path)
cur = conn.cursor()
data = csv.read_db(cur, 'xrp')

input_size = 4
trainig_sets = 3
num_iterations = 2000
learning_rate = 0.5

timerow = get_timerow(data, 'close')
ma = get_ma(timerow, 3)
minimum, maximum = min(ma), max(ma)
normalized_ma = normalize_vals(ma, minimum, maximum)

W = init_weights(input_size)
b = 0

X = np.array([normalized_ma[i:i+input_size] for i in range(trainig_sets)])
Y = np.array([normalized_ma[i+input_size] for i in range(trainig_sets)])

W, b = optimize(W, b, X, Y, learning_rate, num_iterations)
print W.shape

vals = np.array(ma[len(ma)-input_size:])
print vals.shape

prediction = predict(W, vals, b)


print unwrap_prediction(prediction, minimum, maximum)


track_end(start)