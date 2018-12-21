from datetime import timedelta
from datetime import datetime as dt
import numpy as np
from mnist_loader import load_data
import matplotlib.image as img
from matplotlib import cm


target_number = 8
trainig_sets = 100
num_iterations = 1000
learning_rate = 0.1

def track_start():
    start = dt.now()
    print "start time: %s" % str(start.strftime('%Y-%m-%d %H:%M:%S'))
    return start

def track_end(start):
    end = dt.now()
    delta = end - start
    print "end time is %s" % end.strftime('%Y-%m-%d %H:%M:%S')
    print "total duration: %s" % (str(delta-timedelta(microseconds=delta.microseconds)))

