# 실습
# categorical_crossentropy 와 sigmoid 조합

import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. 데이터 구성

x_train = np.load('./_save/_npy/k59_8_train_x.npy')
y_train = np.load('./_save/_npy/k59_8_train_y.npy')
x_test = np.load('./_save/_npy/k59_8_test_x.npy')
y_test = np.load('./_save/_npy/k59_8_test_y.npy')


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(150,150,3)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu')) 
model.add(MaxPooling2D()) 
model.add(Flatten())
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(x_train, y_train)
hist = model.fit(x_train, y_train, epochs=10, validation_split=0.1, batch_size=16
) # 160/5 = 32

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위에거로 시각화 할 것

print('acc :', acc)
print('val_acc: ', val_acc[:-1])

# 4. 평가, 예측

acc = model.evaluate(x_test, y_test)[1]
print('acc : ', acc)



