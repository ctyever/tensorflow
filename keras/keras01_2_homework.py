from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


# 완성한뒤, 출력결과 스샷

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

model = Sequential()
model.add(Dense(1, input_dim=1))

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=1)

loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([6])
print('6의 예측값 : ', result)

# batch_size=2
# loss :  0.3831513822078705
# 6의 예측값 :  [[5.765322]]

# batch_size=1
# loss :  0.3826059103012085
# 6의 예측값 :  [[5.780291]]






