import streamlit as st
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

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

# display 5 head datapoint
st.header("WINE DATASET")
wine_df = pd.DataFrame(wine["data"])
wine_df.columns = wine["feature_names"]
st.write(wine_df.head(5))


st.write(wine)




    