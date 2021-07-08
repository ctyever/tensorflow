import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([range(10)])
x = np.transpose(x)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 
              1.6, 1.5, 1.4, 1.3],
              [10,9,8,7,6,5,4,3,2,1]])
y = np.transpose(y)
# print(x.shape) # 

#완성하시오
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1)) 
model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=500, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict(x)
print('[0] 예측값 : ', y_predict)
print(x)

plt.scatter(x,y[:,0])
plt.scatter(x,y[:,1])
plt.scatter(x,y[:,2])

plt.plot(x, y_predict, color='red')
plt.show()


# epochs=1000, loss :  0.0053992546163499355  [0] 예측값 : [[ 1.006482   1.1345606 10.002953 ]]
# epochs=500, loss :  0.005557237192988396  [0] 예측값 :  [[0.998092  1.1306511 9.999685 ]]