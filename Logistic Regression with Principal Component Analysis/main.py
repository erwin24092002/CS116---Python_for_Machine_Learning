import streamlit as st
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import cv2
from sklearn import decomposition

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
X = np.array(wine["data"])
y = np.array(wine["target"])
y_label = np.array([wine["target_names"]])

# display 5 head datapoint
st.header("WINE DATASET")
wine_df = pd.DataFrame(wine["data"])
wine_df.columns = wine["feature_names"]
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

# CHOSE TYPE OF SPLITTING DATA
cols = st.columns(2)
with cols[0]:
    st.header("Type of Splitting Data")
    split_type = st.selectbox(" ", ("Train-Test Split", "K-Fold Cross Validation"), label_visibility="collapsed")

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

# CHOSE PCA
st.header("Principal Component Analysis")
cols = st.columns(2)
with cols[0]:
    st.write("Status")
    use_pca = st.selectbox(" ", ("Used", "Not Used"), label_visibility="collapsed")
with cols[1]:
    if use_pca == "Used":
        st.write("Number of Components")
        n_components = st.selectbox(" ", range(1, y_label.shape[1]), label_visibility="collapsed")
        pca = decomposition.PCA(n_components=n_components)
    
