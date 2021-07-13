# 06_R2_2를 카피
# 함수형으로 리폼
# 서머리로 확인

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 데이터 구성
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

# 2. 모델 구성

# model = Sequential()
# model.add(Dense(5, input_dim=1))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(2))
# model.add(Dense(1))

input1 = Input(shape=(1, ))
dense1 = Dense(5)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(3)(dense2)
dense4 = Dense(2)(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs= input1, outputs=output1)
model.summary()


model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=3)

loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('x의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)

print("r2스코어 : ", r2)