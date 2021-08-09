# 실습
# categorical_crossentropy 와 sigmoid 조합

import numpy as np 


# 1. 데이터 구성

x_train = np.load('./_save/_npy/k59_8_train_x.npy')
y_train = np.load('./_save/_npy/k59_8_train_y.npy')
x_test = np.load('./_save/_npy/k59_8_test_x.npy')
y_test = np.load('./_save/_npy/k59_8_test_y.npy')


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.10,
    shear_range=0.5,
    fill_mode='nearest'
)

# # 1. ImageDataGenerator 를 정의
# # 2. 파일에서 땡겨올려면 -> flow_from_directory() // x, y가 튜플 형태로 뭉쳐있어
# # 3. 데이터에서 땡겨올려면 -> flow()              // x, y가 나눠있어
# print(x_train.shape, x_test.shape) # (8005, 150, 150, 3) (2023, 150, 150, 3)
# print(np.unique(y_train))

augment_size = 1600

randix = np.random.randint(x_train.shape[0], size=augment_size)


x_augmented = x_train[randix].copy()
y_augmented = y_train[randix].copy()
# print(x_augmented.shape)


x_augmented = x_augmented.reshape(x_augmented.shape[0], 150, 150, 3)
x_train = x_train.reshape(x_train.shape[0], 150, 150, 3)
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3)

import time
start_time = time.time()
x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                batch_size=augment_size, shuffle=False,
                                save_to_dir='d:/temp/' # 이번 파일은 얘가 주인공
                                ).next()[0]
end_time = time.time() - start_time


x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# print(x_train.shape, y_train.shape)  # (9605, 150, 150, 3) (9605,)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) 

x_train = x_train.reshape(9605, 150 * 150 * 3)
x_test = x_test.reshape(2023, 150 * 150 * 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(9605,150,150,3)
x_test = x_test.reshape(2023,150,150,3)


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
model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['acc'])

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

'''
acc :  0.49975284934043884
'''


