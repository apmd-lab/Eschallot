import numpy as np

def read_txt(filename):
    with open(filename + '.txt', 'r') as f:
        labels = f.readline()
        row = f.readline()
        n_col = np.size(row.split())
        line_count = 1
        for line in f:
            line_count += 1
        f.seek(0)
        f.readline()
        raw = np.zeros((line_count, n_col))
        count = 0
        for line in f:
            temp_array = np.array([line.split()])
            raw[count,:] = np.asfarray(temp_array, float)
            count += 1
        
        return raw
