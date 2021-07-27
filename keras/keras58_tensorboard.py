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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', input_shape=(28,28, 1)))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(20, (2,2), activation='relu')) 
model.add(MaxPooling2D()) 
model.add(Flatten()) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.summary()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                    write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=20, batch_size=150, callbacks=[es, tb], validation_split=0.1, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])



'''
tensorboard 
loss :  0.058057837188243866
accuracy :  0.9879999756813049


sparse_categorical_crossentropy
loss :  0.07458729296922684
accuracy :  0.9876000285148621

loss :  0.12132221460342407
accuracy :  0.984499990940094

minmax 처리, batch_size=32 -> batch_size=150
loss :  0.0800686776638031
accuracy :  0.9868000149726868

'''