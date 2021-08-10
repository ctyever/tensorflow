from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd

datasets = pd.read_csv('./data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)

datasets = datasets.to_numpy()
# print(datasets)

x = datasets[:, :11]
y = datasets[:, 11:]
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

# print(y_test)
# print(y_train)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성

# model = LinearSVC()
# accuray_score :  0.5319727891156463
# model = SVC()
# accuray_score :  0.5340136054421769
# model = KNeighborsClassifier()
# accuray_score :  0.5435374149659864
# model = LogisticRegression()
# accuray_score :  0.5231292517006803
# model = DecisionTreeClassifier()
# accuray_score :  0.5891156462585034
model = RandomForestClassifier()
# accuray_score :  0.6503401360544218

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test) # 어큐러씨가 나옴
print('model.score : ', result)

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuray_score : ', acc)


# 딥러닝
# # # loss :  1.0831924676895142
# # # accuracy :  0.5401360392570496