from turtle import xcor
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


st.title("LINEAR REGRESSION WITH STREAMLIT")
st.markdown(
    """
    ### Sinh viên: TRƯƠNG THÀNH THẮNG
    ### MSSV: 20521907
    """
)

st.header("Choose a file")
uploaded_file = st.file_uploader("Choose a file", label_visibility="collapsed")
if uploaded_file is not None:
    # UPLOAD DATA
    bytes_data = uploaded_file.getvalue()
    with open('./'+uploaded_file.name, "wb") as f: 
        f.write(bytes_data)
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # SELECT FEATURES TO TRAIN
    features = df.columns.to_numpy()
    input_features = []
    st.header("Input Features")
    cols = st.columns(4)
    for i in range(len(features)):
        cbox = cols[int(i/len(features)*4)].checkbox(features[i])
        if cbox:
            input_features.append(features[i])

    # CHOSE TYPE OF SPLITTING DATA
    st.header("Type of Splitting Data")
    split_type = st.selectbox(" ", ("Train-Test Split", "K-Fold Cross Validation"), label_visibility="collapsed")
    cols = st.columns(2)
    with cols[0]:
        st.header("Output Feature")
        output_feature = st.selectbox(" ", 
            label_visibility='collapsed',
            options = [feature for feature in features if feature not in input_features and df.dtypes[feature] != 'object'])
    
    # PREPROCESS DATA
    encs = []
    Y = df[output_feature].to_numpy()
    X = np.array([])
    enc_idx = -1  
    for feature in input_features:
        x = df[feature].to_numpy().reshape(-1, 1)
        if (df.dtypes[feature] == 'object'):
            encs.append(OneHotEncoder(handle_unknown='ignore'))
            enc_idx += 1
            x = encs[enc_idx].fit_transform(x).toarray()
        if len(X)==0:
            X = x
        else:
            X = np.concatenate((X, x), axis=1)

    # SET RATIO TO SPLIT DATA
    with cols[1]:
        if split_type == "Train-Test Split":
            st.header("Train Ratio")
            data_ratio = st.slider(label='Select a range of values', 
                label_visibility='collapsed', 
                min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        else: 
            st.header("Numbers of Fold")
            k_fold = st.selectbox(" ", range(2, X.shape[0]), label_visibility="collapsed")

    # TRAIN MODEL
    st.header("Train Model")    
    cols = st.columns(11)
    with cols[5]: 
        btn_run = st.button("Run")
    if btn_run and split_type=="Train-Test Split":
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=data_ratio, random_state=1907)
        model = LinearRegression().fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_true=Y_test, y_pred=Y_pred)
        mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)
        plt.figure(figsize=(8, 4))
        ax1 = plt.subplot()
        ax1.bar(np.arange(1) - 0.21, [mae], 0.4, label='MAE', color='maroon')
        plt.xticks(np.arange(1), [str(data_ratio)])
        plt.xlabel("Folds", color='blue')
        plt.ylabel("Mean Absolute Error", color='maroon')
        ax2 = ax1.twinx()
        ax2.bar(np.arange(1) + 0.21, [mse], 0.4, label='MSE', color='green')
        plt.ylabel('Mean Squared Error', color='green')
        plt.title("EVALUATION METRIC")
        plt.savefig('chart.png')
        with open('model.pkl','wb') as f:
            pickle.dump(model, f)
    elif btn_run: 
        kf = KFold(n_splits=k_fold, random_state=None)
        folds = [str(fold) for fold in range(1, k_fold+1)]
        mae = []
        mse = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index, :], X[test_index, :]
            Y_train, Y_test = Y[train_index], Y[test_index]
            model = LinearRegression().fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            mae.append(round(mean_absolute_error(y_true=Y_test, y_pred=Y_pred), 2))
            mse.append(round(mean_squared_error(y_true=Y_test, y_pred=Y_pred), 2))
            with open('model.pkl','wb') as f:
                pickle.dump(model, f)
        plt.figure(figsize=(8, 4))
        ax1 = plt.subplot()
        ax1.bar(np.arange(len(folds)) - 0.21, mae, 0.4, label='MAE', color='maroon')
        plt.xticks(np.arange(len(folds)), folds)
        plt.xlabel("Folds", color='blue')
        plt.ylabel("Mean Absolute Error", color='maroon')
        ax2 = ax1.twinx()
        ax2.bar(np.arange(len(folds)) + 0.21, mse, 0.4, label='MSE', color='green')
        plt.ylabel('Mean Squared Error', color='green')
        plt.title("EVALUATION METRIC")
        plt.savefig('chart.png')
    img = cv2.imread('chart.png')
    if img is not None: 
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))     

    # SET INPUT
    st.header("Test Model")    
    cols = st.columns(3)
    input = np.array([])
    enc_idx = -1
    for i in range(len(input_features)):
        if (df.dtypes[input_features[i]] == 'object'):
            x = cols[int(i/len(input_features)*3)].selectbox(input_features[i], df[input_features[i]].unique())
            enc_idx += 1 
            x = encs[enc_idx].transform([[x]]).toarray()
        else: 
            x = cols[int(i/len(input_features)*3)].text_input(input_features[i], 0)
            x = np.array([[float(x)]])
        if len(input) == 0: 
            input = x 
        else:
            input = np.concatenate((input, x), axis=1)

    # TEST MODEL
    cols = st.columns(9)
    with cols[4]: 
        btn_predict = st.button("Predict")
    pred_val = [0]
    if btn_predict: 
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
            pred_val = model.predict(input)
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Actual Value")
        st.text_input(" ", label_visibility='collapsed')
    with cols[1]: 
        st.subheader("Predict Value")
        st.text_input(" ", value=round(pred_val[0],2), label_visibility='collapsed', key=2)

        
            

    