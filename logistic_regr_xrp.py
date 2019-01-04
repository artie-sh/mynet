from database_writer import CsvReader
from vectorization_working import *

csv = CsvReader()

conn = csv.connect_db(csv.db_path)
cur = conn.cursor()

data = csv.read_db(cur, 'xrp')

def avg(timerow):
    return sum([line[1] for line in timerow])/max(len(timerow), 1)

def get_timerow(data, rowname='close'):
    rows = {'open': 1, 'high': 2, 'low': 3, 'close': 4}
    colnum = rows[rowname]
    return [(line[0], line[colnum]) for line in data]

def get_ma(datarow, periods):
    return [(datarow[i + periods - 1][0], avg(datarow[i:i + periods])) for i in range(len(datarow) - periods + 1)]

def get_min_max(timerow):
    minimum, maximum = timerow[0][1], timerow[0][1]
    for line in timerow:
        if line[1] < minimum:
            minimum = line[1]
        if line[1] > maximum:
            maximum = line[1]
    return minimum, maximum

def normalize_vals(data, min, max):
    return (data - min) / (max - min)

start = track_start()

timerow = get_timerow(data)
ma = get_ma(timerow, 3)
minimum, maximum = get_min_max(timerow)

input_size = 100
trainig_sets = 1000
num_iterations = 5000
learning_rate = 0.2

W = init_weights(input_size)
b = 0
X = normalize_vals(np.array([[item[1] for item in piece] for piece in [ma[i:i+input_size] for i in range(trainig_sets)]]), minimum, maximum)
Y = normalize_vals(np.array([ma[i+input_size+1][1] for i in range(trainig_sets)]),  minimum, maximum)

W, b = optimize(W, b, X, Y, learning_rate, num_iterations)


track_end(start)