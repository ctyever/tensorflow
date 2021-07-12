import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time


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

start = time.time()
model.fit(x, y, epochs=1000, batch_size=1, verbose=3)
end = time.time() - start
print('걸린시간 : ', end)

# verbose=1
# batch 1, 10 일 때 시간

#4. 평가 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
y_predict = model.predict(x)
print('[10, 1.3, 1] 예측값 : ', y_predict)
# print(x.shape)

# plt.scatter(x[:,0],y)
# plt.scatter(x[:,1],y)
# plt.scatter(x[:,2],y)

# plt.plot(x, y_predict, color='red')
# plt.show()

# verbose
# 0 걸린시간 :  8.92692494392395
# 1 걸린시간 :  14.881126642227173
# 2 걸린시간 :  10.412045955657959
# 3 걸린시간 :  10.339213132858276
