from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
from xgboost import XGBClassifier

# 1. 데이터

datasets = pd.read_csv('./data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)

# # print(datasets.head())
# # print(datasets.shape) # (4898, 12)
# # print(datasets.describe())

# datasets = datasets.values
# # print(type(datasets)) # <class 'numpy.ndarray'>
# # print(datasets.shape)
# x = datasets[:, :11]
# y = datasets[:, 11]

# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.model_selection import train_test_split

# x_train, y_train, x_test, y_test = train_test_split(
#     x, y, random_state=66, shuffle=True, train_size=0.8)

import matplotlib.pyplot as plt
# datasets 의 바 그래프를 그리시오!
# y데이터의 라벨당 갯수를 bar 그래프로 그리시오

count_data = datasets.groupby('quality')['quality'].count()
print(count_data)

'''
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
Name: quality, dtype: int64
'''

# count_data.plot()
plt.bar(count_data.index, count_data)
plt.show()

