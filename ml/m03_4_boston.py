from sklearn.svm import LinearSVC, SVC # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

#1. loss 와 r2로 평가
# MINMax 와 Standard 결과를 명시

from icecream import ic
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn import datasets
from sklearn.datasets import load_boston

# 1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# ic(x.shape, y.shape)  # x.shape: (506, 13), y.shape: (506,)
# ic(datasets.feature_names) # datasets.feature_names: array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')
# ic(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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
# r2_score :  0.7896691868014629
# model = DecisionTreeRegressor()
# r2_score :  0.6729613530651961
model = RandomForestRegressor()
# r2_score :  0.8433246084823727

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test) # r2가 나옴
print('model.score : ', result)

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2_score : ', r2)


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


