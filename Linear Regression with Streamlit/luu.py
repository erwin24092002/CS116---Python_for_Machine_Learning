from asyncore import write
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_test_split_phase(df, selected_columns, y_df):
    y = y_df.to_numpy().reshape(-1, 1)
    X = np.array([])

    enc = OneHotEncoder(handle_unknown='ignore')    
    for column_label in selected_columns:
        col = df[column_label].to_numpy().reshape(-1, 1)
        if (df.dtypes[column_label] == 'object'):
            col = enc.fit_transform(col).toarray()
        if len(X) == 0:
            X = col
        else:
            X = np.concatenate((X, col), axis=1)

    st.header("Train test split: ")
    train_split_ratio = float(st.text_input('Select train ratio in range (0, 1]'))
    if st.button("Train!"):
        if (train_split_ratio and train_split_ratio > 0.0 and train_split_ratio <= 1.0):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split_ratio, random_state=42)
            model = LinearRegression().fit(X_train, y_train)
            return model
        else:
            st.write("Please input train ratio")

    return False

def make_prediction(model, data):
    return

if __name__=='__main__':
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(df)
            column_labels = df.columns.to_numpy()
            selected_columns = []
            st.header("Input feature: ")
            for column_label in column_labels[:-1]:
                t = st.checkbox(column_label)
                if t:
                    selected_columns.append(column_label)
            st.header(f"Output: {column_labels[-1]}")
        
            model = False
            if (len(selected_columns) > 0): 
                model = train_test_split_phase(df, selected_columns, df[column_labels[-1]])
            
            if (model):
                for data_column in selected_columns:
                    st.text_input(data_column)
        except Exception as e:
            st.write(e)
