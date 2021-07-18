from tensorflow.keras.datasets import cifar100
import numpy as np

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
x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) # (50000, 100) (10000, 100)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D()) 
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D()) 
model.add(Flatten()) 
model.add(Dense(128, activation='relu')) 
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=150, callbacks=[es], validation_split=0.05, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

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