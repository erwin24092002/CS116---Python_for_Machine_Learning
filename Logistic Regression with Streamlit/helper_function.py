import matplotlib.pyplot as plt
import numpy as np 

def get_precision(cf_matrix, id): 
    return cf_matrix[id, id] / np.sum(cf_matrix[:, id])

def get_recall(cf_matrix, id): 
    return cf_matrix[id, id] / np.sum(cf_matrix[id, :])

def get_f1(cf_matrix, id):
    pre = get_precision(cf_matrix, id)
    re = get_recall(cf_matrix, id)
    if pre == 0 and re == 0: 
        return 0
    return 2*pre*re/(pre+re)


