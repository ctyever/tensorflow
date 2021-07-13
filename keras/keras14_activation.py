from numpy.testing._private.nosetester import _numpy_tester
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn import datasets 
from sklearn.datasets import load_diabetes

# 1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)
# print(datasets.feature_names)
# print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=10, activation='relu')) #활성화 함수 
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=3, validation_split=0.3, shuffle=True)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
print('예측값 : ', y_predict)

#5. r2 구하기
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)
# epochs=1000, batch_size=1, validation_split=0.3, loss :  3075.999267578125, r2스코어 :  0.5062895077464991
# 1000, 3, 0.3, loss :  3087.773193359375, r2스코어 :  0.5043997328171351