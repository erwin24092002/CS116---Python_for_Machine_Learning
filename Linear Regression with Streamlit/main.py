import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("CAR PRICE")
st.header("Choose a file")
uploaded_file = st.file_uploader("Choose a file", label_visibility="collapsed")
if uploaded_file is not None:
    # Upload data
    bytes_data = uploaded_file.getvalue()
    with open('./'+uploaded_file.name, "wb") as f: 
        f.write(bytes_data)
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Select feature to train
    features = df.columns.to_numpy()
    input_features = []
    st.header("Select input features")
    cols = st.columns(4)
    for i in range(len(features)):
        cbox = cols[int(i/len(features)*4)].checkbox(features[i])
        if cbox:
            input_features.append(features[i])
    cols = st.columns(2)
    with cols[0]:
        st.header("Select output feature")
        output_feature = st.selectbox(" ", 
            label_visibility='collapsed',
            options = [feature for feature in features if feature not in input_features and df.dtypes[feature] != 'object'])
        
    # Ratio to split data
    with cols[1]:
        st.header("Ratio")
        data_ratio = st.slider(label='Select a range of values', 
            label_visibility='collapsed', 
            min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    st.header("Train Model")    
    cols = st.columns(7)
    with cols[3]: 
        btn_run = st.button("Run")
    if btn_run:
        Y = df[output_feature].to_numpy()
        X = np.array([])

        enc = OneHotEncoder(handle_unknown='ignore')    
        for feature in input_features:
            x = df[feature].to_numpy().reshape(-1, 1)
            if (df.dtypes[feature] == 'object'):
                x = enc.fit_transform(x).toarray()
            if len(X)==0:
                X = x
            else:
                X = np.concatenate((X, x), axis=1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=data_ratio, random_state=1907)
        model = LinearRegression().fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_true=Y_test, y_pred=Y_pred)
        mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)
        print(model.score(X_test, Y_test))
        print(mae)
        print(mse) 

    st.header("Test Model")    
    cols = st.columns(3)
    with cols[1]: 
        btn_user = st.button("Specified Input Features")
    if btn_user:
        st.sidebar.title("User Input Feature")
        cols = st.sidebar.columns(2)
        input = []
        for i in range(len(input_features)):
            if (df.dtypes[input_features[i]] == 'object'): 
                input.append(cols[int(i/len(input_features)*2)].selectbox(input_features[i], df[input_features[i]].unique()))
            else: 
                input.append(cols[int(i/len(input_features)*2)].text_input(input_features[i]))
            

    