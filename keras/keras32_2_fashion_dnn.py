from tensorflow.keras.datasets import fashion_mnist
import numpy as np

 # 1. 데이터 구성
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

# print(x_train.shape) # (60000, 28, 28, 1)
# print(x_test.shape) # (10000, 28, 28, 1)

# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Dense(100, input_shape=(28 * 28,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(64, activation='relu')) 
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', input_shape=(28,28, 1)))
# model.add(Conv2D(20, (2,2), activation='relu'))
# model.add(Conv2D(20, (2,2), activation='relu')) 
# model.add(MaxPooling2D()) 
# model.add(Flatten()) 
# model.add(Dense(64, activation='relu')) 
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))


# #3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10, batch_size=32, callbacks=[es], validation_split=0.1, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# loss :  0.9283794164657593
# accuracy :  0.8973000049591064

# dnn
# loss :  0.37769749760627747
# accuracy :  0.8752999901771545





