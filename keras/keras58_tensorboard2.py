from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,6,7,8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) # Dense layer y = wx + b
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

from tensorflow.keras.callbacks import TensorBoard

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                    write_graph=True, write_images=True)

model.fit(x, y, epochs=100, batch_size=1, callbacks=[tb], validation_split=0.1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([6])
print('6의 예측값 : ', result)

# loss :  0.380204975605011
# 6의 예측값 :  [[5.675434]]