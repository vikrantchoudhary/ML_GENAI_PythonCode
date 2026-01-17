# a basic code to get min max scaling to transfer data  in range [0,1] using numpy
import numpy as np

def min_max_scalar(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)


features = np.array([10,12,16,24,18,40,16,27,31]);
print(min_max_scalar(features))
