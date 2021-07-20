import numpy as np  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(4,3,1) # ( batch_size, timesteps, feature)

# 2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
model.add(SimpleRNN(10, activation='relu', input_length=3, input_dim=1)) # 이렇게 써줄 수도 있음
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

# model.summary()

#  total params = ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1 * unit 개수)

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


# [[8.729839]]

# 모델 변경, [[8.114725]]

# 모델 변경 [[8.038197]]

# validation_split=0.1, earlystopping 추가
# [[7.9194455]]

# validation 제거
# [[7.9818273]]

# [[8.003282]]

