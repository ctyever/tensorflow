from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn import datasets  
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(y[:100])

# print(x.shape, y.shape) # (569, 30) (569,)

# print(np.unique(y)) # (0, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성

# model = LinearSVC()
# accuray_score :  0.9766081871345029
# model = SVC()
# accuray_score :  0.9824561403508771
# model = KNeighborsClassifier()
# accuray_score :  0.9649122807017544
# model = LogisticRegression()
# accuray_score :  0.9766081871345029
# model = DecisionTreeClassifier()
# accuray_score :  0.9415204678362573
model = RandomForestClassifier()
# accuray_score :  0.9707602339181286

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test) # 어큐러씨가 나옴
print('model.score : ', result)

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuray_score : ', acc)

