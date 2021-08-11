from sklearn.svm import LinearSVC, SVC # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from icecream import ic
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn import datasets
from sklearn.datasets import load_diabetes

import warnings

warnings.filterwarnings('ignore')

# 1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# ic(x.shape, y.shape)  # x.shape: (506, 13), y.shape: (506,)
# ic(datasets.feature_names) # datasets.feature_names: array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')
# ic(datasets.DESCR)


# 2. 모델구성

# model = LinearSVC()
# ValueError: Unknown label type: 'continuous'
# model = SVC()
# ValueError: Unknown label type: 'continuous'
# model = KNeighborsClassifier()
# ValueError: Unknown label type: 'continuous'
# model = LogisticRegression()
# ValueError: Unknown label type: 'continuous'
# model = LinearRegression()
# r2:  [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 0.4876
# model = DecisionTreeRegressor()
# r2:  [-0.29282795 -0.14707361 -0.01416748 -0.11120162  0.02630176] -0.1078
model = RandomForestRegressor()
# r2:  [0.37461763 0.49603705 0.47690545 0.39319224 0.43097139] 0.4343

#3. 컴파일, 훈련
#4. 평가, 예측
scores = cross_val_score(model, x, y, cv=kfold, scoring='r2')

print("r2: ", scores, round(np.mean(scores), 4))


# 딥러닝
# MinMaxScaler, batch_size=32
# loss :  11.4025239944458
# r2스코어 :  0.8715382774870997
# MinMaxScaler, batch_size=8, validation_split=0.08
# loss :  11.118603706359863
# r2스코어 :  0.8747369566931819
# MinMaxScaler, batch_size=8, validation_split=0.3, random_state=9
# loss :  9.450647354125977
# r2스코어 :  0.8935282796100319


