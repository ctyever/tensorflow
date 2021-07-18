import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 
              1.6, 1.5, 1.4, 1.3]])
# print(x.shape) # (2, 10)

x = np.transpose(x)
# x2 = x.swapaxes(0, 1) 다른 방법
# print(x.shape) # (10, 2)
# print(x)

'''
[[ 1.   1. ]
 [ 2.   1.1]
 [ 3.   1.2]
 [ 4.   1.3]
 [ 5.   1.4]
 [ 6.   1.5]
 [ 7.   1.6]
 [ 8.   1.5]
 [ 9.   1.4]
 [10.   1.3]]
'''

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# print(y.shape) # (10,)

#완성하시오
#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=2)) 
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=100, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict(x)
print('예측값 : ', y_predict)

# epochs=500, loss :  0.10875537246465683  [10, 1.3] 예측값 :  [[19.549253]]
# epochs=1000, loss :  0.012192628346383572 [10, 1.3] 예측값 :  [[19.791632]] / loss :  0.0016621561953797936 [10, 1.3] 예측값 :  [[19.968134]]
# epochs=1500, loss :  2.7036639593802647e-08 [10, 1.3] 예측값 :  [[19.999897]]

plt.scatter(x[:,0],y)
plt.scatter(x[:,1],y)

plt.plot(x, y_predict, color='red')
plt.show()



