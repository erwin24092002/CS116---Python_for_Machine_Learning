import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def addlabels(x, y, color):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center', fontsize=10, bbox = dict(facecolor = color, alpha =.6))

def plot_bar(x_data, y_data, color, x_label, y_label, title):
    plt.bar(x_data, y_data, color = color, width = 0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim((0, max(y_data)*1.2))
    addlabels(x_data, y_data, color)
    plt.title(title)