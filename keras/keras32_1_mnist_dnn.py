import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)
# 전처리 해야겠지???

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# print(np.min(x_train), np.max(x_train)) # 0 255
x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape)

# print(x_train.shape)
# print(x_test.shape)

# print(np.unique(y_train))

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# # scaler = QuantileTransformer()
# # scaler = PowerTransformer()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D

model = Sequential()
# model.add(Dense(100, input_shape=(28 * 28,)))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(64, activation='relu')) 
# model.add(Dense(64, activation='relu'))
# # model.add(GlobalAveragePooling2D())
# model.add(Dense(10, activation='softmax'))

model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', input_shape=(28,28, 1)))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(20, (2,2), activation='relu')) 
model.add(MaxPooling2D()) 
# model.add(Flatten()) 
# model.add(Dense(64, activation='relu')) 
# model.add(Dense(32, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=150, callbacks=[es], validation_split=0.1, verbose=2)
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# plt 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')


# 2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

# loss :  0.12132221460342407
# accuracy :  0.984499990940094

# minmax 처리, batch_size=32 -> batch_size=150
# loss :  0.0800686776638031
# accuracy :  0.9868000149726868

# dnn 
# loss :  0.1577836275100708
# accuracy :  0.9747999906539917

# gap 처리
# loss :  0.29182305932044983
# accuracy :  0.916100025177002