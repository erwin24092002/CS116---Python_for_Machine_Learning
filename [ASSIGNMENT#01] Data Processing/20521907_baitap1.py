#                 ASSIGNMENT#01 - DATA PROCESSING

from sklearn import datasets 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Tải dữ liệu 
iris = datasets.load_iris()
X = iris.data[:, :]
y = iris.target


# 2. Chia dữ liệu huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_predict = (np.random.rand(y_test.shape[0])*3).astype(int)

def visualize_result(y_test, y_predict, title):
    fig, ax = plt.subplots(figsize=(10,4))
    cm = confusion_matrix(y_test, y_predict)
    sns.heatmap(data = cm, cmap="Blues",
            annot=True, fmt=".2f",
            linecolor='white', linewidths=0.5);
    yticks = ['Setosa', 'Versicolour', 'Virginica']
    xticks = ['Setosa', 'Versicolour', 'Virginica']
    ax.set_yticklabels(yticks, rotation=0);
    ax.set_xticklabels(xticks, rotation=0);
    ax.set_xlabel('Groundtruth', color='red')
    ax.set_ylabel('Predict', color='red')
    ax.set_title(title, color='red')
    plt.show()

visualize_result(y_test, y_predict, 'RANDOM PREDICTION ON TESTSET')


# 3. Chuẩn hoá dữ liệu (Standard Score) 
def normalize_data(X_train, X_test):
    for i in range(X_train.shape[1]):
        mean = np.mean(X_train[:, i])
        var = np.var(X_train[:, i])
        sigma = np.sqrt(var)

        X_train[:, i] = (X_train[:, i] - mean) / sigma 
        X_test[:, i] = (X_test[:, i] - mean) / sigma

normalize_data(X_train, X_test)


# 4. Trực quan hoá dữ liệu
from turtle import tilt


def visualize_data(X_, y_, title):
    plt.subplot(1, 2, 1)
    plt.scatter(X_[:, 0], X_[:, 1], c=y_, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.title('SEPAL ON ' + title, color='red')

    plt.subplot(1, 2, 2)
    plt.scatter(X_[:, 0], X_[:, 1], c=y_, cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal width")
    plt.title('PETAL  ON ' + title, color='red')

    plt.show()    
    return 

visualize_data(X_train, y_train, 'TRAIN SET')
visualize_data(X_test, y_test, 'TEST SET')

