from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D

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
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. 모델 구성


model = Sequential()
# model.add(LSTM(units=10, activation='relu', input_shape=(28 * 28, 1)))
model.add(Conv1D(128, 2, activation='relu', input_shape=(28 * 28, 1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# #3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                        filepath='./_save/ModelCheckPoint/keras48_MCP_fashion.hdf5')


model.fit(x_train, y_train, epochs=100, batch_size=1500, callbacks=[es, cp], validation_split=0.1, verbose=2)

# model.save('./_save/ModelCheckPoint/keras48_model_fashion_save.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_model_fashion_save.h5') # save_model
# model = load_model('./_save/ModelCheckPoint/keras48_MCP_fashion.hdf5') #체크포인트

#4. 평가, 예측


loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# checkpoint
# loss :  0.31809720396995544
# accuracy :  0.8885999917984009

# load
# loss :  0.31905508041381836
# accuracy :  0.8889999985694885

# conv1d
# loss :  0.3996327817440033
# accuracy :  0.8870000243186951

# lstm
# loss :  nan
# accuracy :  0.10000000149011612

# loss :  0.9283794164657593
# accuracy :  0.8973000049591064

# dnn
# loss :  0.37769749760627747
# accuracy :  0.8752999901771545





