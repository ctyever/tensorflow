# overfit을 극복하자!!!

# 1. 젠체 훈련을 데이터가 마니 마니!!!
# 2. 노말라이제이션 (normaliztion) layer에서도 노말라이제이션 한다?? 
# Fully connected layer 노드가 많을 때 훈련의 과적합이 생길 수 있음
# 3. dropout

from tensorflow.keras.datasets import cifar100
import numpy as np

# 데이터 과적합에 대해서 생각해보기

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

# print(np.unique(y_train)) 

'''
 [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
 96 97 98 99]
 '''
# print(np.min(x_train), np.max(x_train)) # 0 255
# x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
# x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

x_train = x_train.reshape(50000, 32 * 32 * 3)/255.
x_test = x_test.reshape(10000, 32 * 32 * 3)/255.
# 2차원으로 reshpae 하고 다시 4차원으로 원위치
# print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()
scaler = StandardScaler()
# x_train = scaler.fit(x_train)
# x_train = scaler.transform(x_train)
x_train = scaler.fit_transform(x_train) # 한번에 써줄 수 있음, train 에서만 쓴다
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(32, 32, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D()) 

model.add(Conv2D(64, (2, 2), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu')) 
model.add(MaxPooling2D()) 

model.add(Flatten()) 
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es], validation_split=0.25, verbose=2)
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time )
print('loss : ', loss[0])
print('acc : ', loss[1])

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


# loss :  10.989113807678223
# accuracy :  0.2142000049352646

#layer 변경
# loss :  9.050128936767578
# accuracy :  0.33500000834465027

# validation_split=0.1 -> 0.3
# loss :  9.38029956817627
# accuracy :  0.3127000033855438

# validation_split=0.1, patience=10
# loss :  9.958231925964355
# accuracy :  0.32510000467300415

# validation_split=0.08, epochs=100, patience=10
# loss :  9.745123863220215
# accuracy :  0.33899998664855957

# epochs=1000
# loss :  11.141669273376465
# accuracy :  0.33410000801086426

# validation_split=0.05
# loss :  10.066449165344238
# accuracy :  0.3555000126361847

# validation_split=0.01
# loss :  9.421782493591309
# accuracy :  0.34940001368522644

# 걸린 시간 :  187.5792374610901
# loss :  9.897117614746094
# accuracy :  0.33489999175071716

# standard 
# 걸린 시간 :  176.1574444770813
# loss :  9.932455062866211
# acc :  0.33899998664855957

# batch 150 -> 256, validation 0.05 -> 0.2

# batch_size=64

# validation_split=0.25, monitor='val_loss' / 모델 수정
# 걸린 시간 :  107.7505042552948
# loss :  5.421496868133545
# acc :  0.32179999351501465

# dropout 
# 걸린 시간 :  201.7531008720398
# loss :  2.586777925491333
# acc :  0.3700000047683716

