import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# df = pd.read_csv("car_price.csv")
# features = ['fueltype','carlength','carwidth']
# # features = ['carlength','carwidth']
# Y = df['price'].to_numpy()
# X = np.array([])
# encs = []
# enc_idx = -1
# for feature in features:
#     x = df[feature].to_numpy().reshape(-1, 1)
#     if (df.dtypes[feature] == 'object'):
#         encs.append(OneHotEncoder(handle_unknown='ignore'))
#         enc_idx += 1
#         x = encs[enc_idx].fit_transform(x).toarray()
#     if len(X)==0:
#         X = x
#     else:
#         X = np.concatenate((X, x), axis=1)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=1907)
# model = LinearRegression().fit(X_train, Y_train)
# Y_pred = model.predict(X_test)
# score = model.score(X_test, Y_test)
# mae = mean_absolute_error(y_true=Y_test, y_pred=Y_pred)
# mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)
# print(score)
# print(mae)
# print(mse)

# x_test = encs[0].transform(['gas']).toarray().reshape(-1, 1)
# print(X_test[0])
# print(x_test)
# x_test = np.concatenate((x_test, np.array([[160.0]])), axis=1)
# x_test = np.concatenate((x_test, np.array([[60.0]])), axis=1)


# # print(X_train[0])
# # x_test = np.array([[160.0, 60.0]])
# print(x_test)
# print(model.predict(x_test))

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    print(model.predict(np.array([[0., 1., 160., 80., 160.]])))