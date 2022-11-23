import streamlit as st
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import cv2
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, log_loss, f1_score
import pickle
from support_function import *
import math

st.title("LOGISTIC REGRESSION WITH PRINCIPAL COMPONENT ANALYSIS")
st.markdown(
    """
    ### Sinh viên: TRƯƠNG THÀNH THẮNG
    ### MSSV: 20521907
    """
)

print("-----------------------------REFRESHED-------------------------")
# wine image header
wine_img = cv2.imread('wine_image.jpg')
wine_img = cv2.cvtColor(wine_img, cv2.COLOR_BGR2RGB)
st.image(wine_img, width=700)

# load wine dataset
wine = load_wine()
y_label = wine["target_names"]

# display 5 head datapoint
st.header("WINE DATASET")
wine_df = pd.DataFrame(wine["data"])
wine_df.columns = wine["feature_names"]
wine_df["target"] = wine["target"]
wine_df = wine_df.sample(frac=1) # suffling data point
st.write(wine_df.head(5))

# select feature to predict
st.header("Input Features")
features = np.array(wine["feature_names"])
input_features = []
cboxs = []
cols = st.columns(3)
for i in range(len(features)):
    cboxs.append(cols[i%3].checkbox(" ".join(features[i].split('_'))))
    if cboxs[i]:
        input_features.append(features[i])

# pre-processing data
y = np.array(np.array(wine_df["target"]))
X = np.array([])
for feature in input_features:
    x = wine_df[feature].to_numpy().reshape(-1, 1)
    if len(X)==0:
        X = x
    else:
        X = np.concatenate((X, x), axis=1)

# chose type of splitting data
cols = st.columns(2)
with cols[0]:
    st.header("Type of Splitting Data")
    split_type = st.selectbox(" ", ("Train-Test Split", "K-Fold Cross Validation"), label_visibility="collapsed")
with cols[1]:
    if split_type == "Train-Test Split":
        st.header("Train Ratio")
        data_ratio = st.slider(label='Select a range of values', 
            label_visibility='collapsed', 
            min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    else: 
        st.header("Numbers of Fold")
        k_fold = st.selectbox(" ", range(2, X.shape[0]), label_visibility="collapsed")

# chose pca
st.header("Principal Component Analysis")
cols = st.columns(2)
with cols[0]:
    st.write("Status")
    use_pca = st.selectbox(" ", ("Used", "Not Used"), label_visibility="collapsed")
with cols[1]:
    if use_pca == "Used":
        st.write("Number of Components")
        n_components = st.selectbox(" ", range(1, 1 if len(input_features)==0 else len(input_features)), label_visibility="collapsed")
        pca = decomposition.PCA(n_components=n_components)

# TRAIN MODEL
st.header("Train Model")    
cols = st.columns(11)
with cols[5]: 
    btn_run = st.button("Run")    

if btn_run and split_type=="Train-Test Split": 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=data_ratio, random_state=1907)
    if use_pca == "Used":
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
    model = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X_train, y_train)
    cf_matrix = confusion_matrix(y_test, model.predict(X_test))
    logloss = log_loss(y_test, model.predict_proba(X_test))
    st.write("Train Ratio: {} --- Log Loss: {:0.3f}".format(data_ratio, logloss))
    cols = st.columns(2)
    with cols[0]:
        visualize_result(cf_matrix, "Confusion Matrix", y_label)
        plt.savefig('images/confusion_matrix.png')
        img = cv2.imread('images/confusion_matrix.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img)
    with cols[1]: 
        plot_performence_chart(cf_matrix, labels=y_label)
        plt.savefig('images/performence_chart.png')
        img = cv2.imread('images/performence_chart.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img)
elif btn_run: 
    kf = KFold(n_splits=k_fold, random_state=None)
    id = 0
    for train_index, test_index in kf.split(X):
        id += 1
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        if use_pca == "Used":
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
        model = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X_train, y_train)
        cf_matrix = confusion_matrix(y_test, model.predict(X_test))
        logloss = log_loss(y_test, model.predict_proba(X_test))
        st.write("Fold: {} --- Log Loss: {:0.3f}".format(id, logloss))
        cols = st.columns(2)
        with cols[0]:
            visualize_result(cf_matrix, "Confusion Matrix", y_label)
            plt.savefig('images/confusion_matrix.png')
            img = cv2.imread('images/confusion_matrix.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img)
        with cols[1]: 
            plot_performence_chart(cf_matrix, labels=y_label)
            plt.savefig('images/performence_chart.png')
            img = cv2.imread('images/performence_chart.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img)

cols = st.columns(5)
with cols[2]: 
    btn_runall = st.button("All Number of Components")    
if btn_runall and split_type=="Train-Test Split": 
    acc = []
    f1 = []
    n = []
    for n_component in range(1, len(input_features)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=data_ratio, random_state=1907)
        pca = decomposition.PCA(n_components=n_component)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        model = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X_train, y_train)
        cf_matrix = confusion_matrix(y_test, model.predict(X_test))
        acc.append(get_acc(cf_matrix))
        f1.append(f1_score(y_test, model.predict(X_test), average='micro'))
        n.append(n_component)
    plt.figure(figsize=(10,4))
    plt.ylim((0, 1.4))
    plt.bar(np.arange(len(np.array(acc))), np.array(acc), 0.4, label='Accuracy', color='green')
    plt.xticks(np.arange(len(np.array(n))), np.array(n))
    plt.xlabel("Number of Components")
    plt.ylabel("Perfomence (%)")
    addlabels(np.array(n), np.array(acc), color='green', denta=0)
    plt.legend()
    plt.savefig("chart.png")
    img = cv2.imread('chart.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img)

    plt.figure(figsize=(10,4))
    plt.ylim((0, 1.4))
    plt.bar(np.arange(len(np.array(f1))), np.array(acc), 0.4, label='F1 score', color='blue')
    plt.xticks(np.arange(len(np.array(n))), np.array(n))
    plt.xlabel("Number of Components")
    plt.ylabel("Perfomence (%)")
    addlabels(np.array(n), np.array(f1), color='blue', denta=0)
    plt.legend()
    plt.savefig("chart.png")
    img = cv2.imread('chart.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img)
    
elif btn_runall: 
    kf = KFold(n_splits=k_fold, random_state=None)
    id = 0
    acc = []
    f1 = []
    n = []
    for n_component in range(1, len(input_features)+1):
        sacc = []
        sf1= []
        for train_index, test_index in kf.split(X):
            id += 1
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            pca = decomposition.PCA(n_components=n_component)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
            model = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X_train, y_train)
            cf_matrix = confusion_matrix(y_test, model.predict(X_test))
            sacc.append(get_acc(cf_matrix))
            sf1.append(f1_score(y_test, model.predict(X_test), average='micro'))
        acc.append(np.mean(np.array(sacc)))
        f1.append(np.mean(np.array(sf1)))
        n.append(n_component)

    plt.figure(figsize=(10,4))
    plt.ylim((0, 1.4))
    plt.bar(np.arange(len(np.array(acc))), np.array(acc), 0.4, label='Accuracy', color='green')
    plt.xticks(np.arange(len(np.array(n))), np.array(n))
    plt.xlabel("Number of Components")
    plt.ylabel("Perfomence (%)")
    addlabels(np.array(n), np.array(acc), color='green', denta=0)
    plt.legend()
    plt.savefig("chart.png")
    img = cv2.imread('chart.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img)

    plt.figure(figsize=(10,4))
    plt.ylim((0, 1.4))
    plt.bar(np.arange(len(np.array(f1))), np.array(acc), 0.4, label='F1 score', color='blue')
    plt.xticks(np.arange(len(np.array(n))), np.array(n))
    plt.xlabel("Number of Components")
    plt.ylabel("Perfomence (%)")
    addlabels(np.array(n), np.array(f1), color='blue', denta=0)
    plt.legend()
    plt.savefig("chart.png")
    img = cv2.imread('chart.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img)