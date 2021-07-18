import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 
              1.6, 1.5, 1.4, 1.3],
              [10,9,8,7,6,5,4,3,2,1]])
# print(x.shape) # 

x = np.transpose(x)
# x2 = x.swapaxes(0, 1) 다른 방법


y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


#완성하시오
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3)) 
model.add(Dense(1))
# model.add(Dense(4))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict(x)
print('x 예측값 : ', y_predict)

plt.scatter(x[:,0],y)
plt.scatter(x[:,1],y)
plt.scatter(x[:,2],y)

plt.plot(x, y_predict, color='red')
plt.show()
# epochs=1000, loss :  5.725305385340107e-08, [10, 1.3, 1] 예측값 :  [[19.999792]]
# epochs=3000, loss :  2.8588983695954084e-06 [10, 1.3, 1] 예측값 :  [[19.999361]]
# epochs=3000, 1layer : 10, loss :  1.957477024916443e-06 [10, 1.3, 1] 예측값 :  [[19.999065]]
# epochs=1000, Dense(10 -> 1)loss :  1.3414137578493524e-09, [10, 1.3, 1] 예측값 :  [[20.000053]]