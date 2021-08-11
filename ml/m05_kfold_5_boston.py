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
from sklearn.datasets import load_boston

import warnings

warnings.filterwarnings('ignore')

# 1.데이터
datasets = load_boston()
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
# r2:  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 0.7128
model = DecisionTreeRegressor()
# r2:  [0.70773775 0.68592681 0.81874738 0.69731847 0.84266107] 0.7505
# model = RandomForestRegressor()
# r2:  [0.92359427 0.84686784 0.81548493 0.88594692 0.89704639] 0.8738

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


