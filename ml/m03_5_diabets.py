from sklearn.svm import LinearSVC, SVC # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

from icecream import ic
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn import datasets 
from sklearn.datasets import load_diabetes

# 1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# ic(x.shape, y.shape)  # x.shape: (442, 10), y.shape: (442,)
# ic(datasets.feature_names)  # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(y)

# 2. 모델구성

# model = LinearSVC()
# r2_score :  0.08361192312476673
# model = SVC()
# r2_score :  0.19635021457892765
# model = KNeighborsClassifier()
# ValueError: Unknown label type: 'continuous'
# model = LogisticRegression()
# ValueError: Unknown label type: 'continuous'
model = LinearRegression()
# r2_score :  0.5900352656383733
# model = DecisionTreeRegressor()
# r2_score :  0.04137486005722102
# model = RandomForestRegressor()
# r2_score :  0.5055226291360866

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test) # 어큐러씨가 나옴
print('model.score : ', result)

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = r2_score(y_test, y_predict)
print('r2_score : ', acc)


# MinMaxScaler
# loss :  1990.69287109375
# r2스코어 :  0.6397765814915889

