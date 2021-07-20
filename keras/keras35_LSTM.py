import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(4,3,1) # ( batch_size, timesteps, feature)

# 2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
model.add(LSTM(units=10, activation='relu', input_shape=(3,1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

# model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 10)                110
_________________________________________________________________
dense_1 (Dense)              (None, 10)                110
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 30
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 6
=================================================================
Total params: 791
Trainable params: 791
Non-trainable params: 0
_________________________________________________________________
'''

# total params  = (INPUT + BIAS + OUTPUT ) * OUTPUT
# LSTM total params = ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1 * unit 개수) * 4

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)

model.fit(x, y, epochs=300, batch_size=1, callbacks=[es])

# 4. 평가, 예측
x_input = np.array([5,6,7]).reshape(1, 3, 1)

# loss = model.evaluate(x, y)
# print('loss : ', loss)
result = model.predict(x_input)
print(result)



