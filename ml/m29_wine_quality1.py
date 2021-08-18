from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
from xgboost import XGBClassifier

# 1. 데이터

datasets = pd.read_csv('./data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)

# print(datasets.head())
# print(datasets.shape) # (4898, 12)
# print(datasets.describe())

datasets = datasets.values
# print(type(datasets)) # <class 'numpy.ndarray'>
# print(datasets.shape)
x = datasets[:, :11]
y = datasets[:, 11]

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8
)

# scale = StandardScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

# # 2. 모델
# model = XGBClassifier(n_jobs=-1)

# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)

# print("accuracy : ", score)




