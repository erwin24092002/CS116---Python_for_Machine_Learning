import matplotlib.pyplot as plt
import numpy as np 

def get_precision(cf_matrix, id): 
    return cf_matrix[id, id] / np.sum(cf_matrix[:, id])

def get_recall(cf_matrix, id): 
    return cf_matrix[id, id] / np.sum(cf_matrix[id, :])

def get_f1(cf_matrix, id):
    pre = get_precision(cf_matrix, id)
    re = get_recall(cf_matrix, id)
    return 2*pre*re/(pre+re)

cf_matrix = np.array([
    [20, 25], 
    [20, 15]
])

print(get_precision(cf_matrix=cf_matrix, id=1))
print(get_recall(cf_matrix=cf_matrix, id=1))
print(get_f1(cf_matrix=cf_matrix, id=1))

