
import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])


# print(x.shape, y.shape) # (13, 3) (13,)

# x = x.reshape(13,3,1) # ( batch_size, timesteps, feature)
x = x.reshape(x.shape[0], x.shape[1], 1)

# 2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
model.add(SimpleRNN(units=32, activation='relu', input_shape=(3,1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

# model.summary()

# total params  = (INPUT + BIAS + OUTPUT ) * OUTPUT
# LSTM total params = ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1 * unit 개수) * 4

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)

model.fit(x, y, epochs=300, batch_size=1,  callbacks=[es])

# 4. 평가, 예측
# x_input = np.array([5,6,7]).reshape(1, 3, 1)

# loss = model.evaluate(x, y)
# print('loss : ', loss)
x_predict = np.array([50,60,70]).reshape(1,3,1)

result = model.predict(x_predict)
print(result)

# GRU
# [[80.282875]]

# simplernn
# [[81.94877]]