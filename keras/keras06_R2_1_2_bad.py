#1. R2 를 음수가 아닌 0.5 이하로 만들어라.
#2. 데이터 건들지 마
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4. epochs는 100 이상
#5. batch_size = 1
#6. 히든레이어의 노드는 10개 이상 1000개 이하
#7. train 70%

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
model.add(Dense(1000, input_dim=1)) 
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
print('x_test의 예측값 : ', y_predict)  

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)
