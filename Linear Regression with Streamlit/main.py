import streamlit as st
import pandas as pd

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