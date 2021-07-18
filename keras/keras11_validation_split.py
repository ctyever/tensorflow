from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])


x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) # Dense layer y = wx + b
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
# result = model.predict([11])
y_predict = model.predict([x])
print('예측값 : ', y_predict)  
# epochs=1000, loss :  2.3646862246096134e-11, 11의 예측값 :  [[10.999997]]

print(x_train)
print(y_train)
print(x_test)
print(y_test)