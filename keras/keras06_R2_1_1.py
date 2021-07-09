from numpy.testing._private.nosetester import _numpy_tester
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(100))
y = np.array(range(1, 101))

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66) # shuffle=True 가 디폴트 / random_state=66 (난수표) random_state의 숫자가 동일하면 동일하게 나옴 /random_state 가급적 써줘라


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) # Dense layer y = wx + b
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
print('x_test의 예측값 : ', y_predict)  
# epochs=1000, loss :  1.6540276703835843e-08, 11의 예측값 :  [[11.999878]]
# epochs=100 , loss :  8.50045671541011e-06, 11의 예측값 :  [[ 8.996602 ]

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)

