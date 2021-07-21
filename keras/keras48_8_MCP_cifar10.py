from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D

# 1. 데이터 구성

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)
# print(np.min(x_train), np.max(x_train)) # 0 255 

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

# print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) # (50000, 10) (10000, 10)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


# 2. 모델 구성

# model = Sequential()
# # model.add(LSTM(units=10, activation='relu', input_shape=(32*32*3, 1)))
# model.add(Conv1D(64, 2, activation='relu', input_shape=(32*32*3, 1)))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))

# # model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', input_shape=(32, 32, 3)))
# # model.add(Conv2D(20, (2,2), activation='relu'))
# # model.add(Conv2D(20, (2,2), activation='relu')) 
# # model.add(MaxPooling2D()) 
# # model.add(Flatten()) 
# # model.add(Dense(64, activation='relu')) 
# # model.add(Dense(32, activation='relu'))
# # model.add(Dense(10, activation='softmax'))

# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                         filepath='./_save/ModelCheckPoint/keras48_MCP_cifar10.hdf5')

# model.fit(x_train, y_train, epochs=100, batch_size=1500, callbacks=[es, cp], validation_split=0.08, verbose=2)

# model.save('./_save/ModelCheckPoint/keras48_model_cifar10_save.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_model_cifar10_save.h5') # save_model
model = load_model('./_save/ModelCheckPoint/keras48_MCP_cifar10.hdf5') #체크포인트


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# checkpoint
# loss :  1.4343268871307373
# accuracy :  0.5008000135421753

# load
# loss :  1.4451375007629395
# accuracy :  0.49889999628067017

# conv1d
# loss :  1.588168978691101
# accuracy :  0.47760000824928284

# lstm
# loss :  nan
# accuracy :  0.10000000149011612

# loss :  3.7575008869171143
# accuracy :  0.6108999848365784

# dnn
# loss :  1.9412628412246704
# accuracy :  0.4837999939918518