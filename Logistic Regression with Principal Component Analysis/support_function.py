import matplotlib.pyplot as plt
import numpy as np 

def get_precision(cf_matrix, id): 
    if np.sum(cf_matrix[:, id]) == 0: 
        return 0
    return cf_matrix[id, id] / np.sum(cf_matrix[:, id])

def get_recall(cf_matrix, id): 
    if np.sum(cf_matrix[id, :]) == 0: 
        return 0
    return cf_matrix[id, id] / np.sum(cf_matrix[id, :])

def get_f1(cf_matrix, id):
    pre = get_precision(cf_matrix, id)
    re = get_recall(cf_matrix, id)
    if pre == 0 and re == 0: 
        return 0
    return 2*pre*re/(pre+re)

def plot_performence_chart(cf_matrix, labels):
    pre = []
    re = []
    f1 = []
    for idex in range(cf_matrix.shape[0]): 
        pre.append(get_precision(cf_matrix, idex))
        re.append(get_recall(cf_matrix, idex))
        f1.append(get_f1(cf_matrix, idex))
    plt.figure(figsize=(8,4))
    plt.ylim((0, 1.2))
    plt.bar(np.arange(len(labels)) - 0.21, pre, 0.2, label='Precision', color='maroon')
    plt.bar(np.arange(len(labels)) , re, 0.2, label='Recall', color='orange')
    plt.bar(np.arange(len(labels)) + 0.21, f1, 0.2, label='F1 Score', color='green')
    plt.xticks(np.arange(len(labels)), labels)
    plt.xlabel("Classes")
    plt.ylabel("Performance (%)")
    plt.legend()