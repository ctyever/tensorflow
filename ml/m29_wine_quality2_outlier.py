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

def outliers(data_out):
    quartile_1, q2, quertile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quertile_3)
    iqr = quertile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quertile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(x_train)

print('이상치의 위치 : ', outliers_loc)
###### 아웃라이어의 갯수를 count 하는 기능 추가 할것

# import matplotlib.pyplot as plt

# plt.boxplot(, sym="bo")
# plt.show()