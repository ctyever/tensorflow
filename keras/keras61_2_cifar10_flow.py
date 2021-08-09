# 훈련데이터를 10만개로 증폭할것!
# 완료 후 기존 모델과 비교
# save_dir 도 temp 에 넣을것

import numpy as np 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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

# 1. ImageDataGenerator 를 정의
# 2. 파일에서 땡겨올려면 -> flow_from_directory() // x, y가 튜플 형태로 뭉쳐있어
# 3. 데이터에서 땡겨올려면 -> flow()              // x, y가 나눠있어
# print(x_train.shape, y_train) # (50000, 32, 32, 3)

augment_size = 50000

randix = np.random.randint(x_train.shape[0], size=augment_size)


x_augmented = x_train[randix].copy()
y_augmented = y_train[randix].copy()
# print(x_augmented.shape)


x_augmented = x_augmented.reshape(x_augmented.shape[0], 32, 32, 3)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

import time
start_time = time.time()
x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                batch_size=augment_size, shuffle=False,
                                save_to_dir='d:/temp/' # 이번 파일은 얘가 주인공
                                ).next()[0]
end_time = time.time() - start_time


x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# print(x_train.shape, y_train.shape)  # (100000, 32, 32, 3) (100000, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) # (100000, 10) (10000, 10)

x_train = x_train.reshape(100000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(100000,32,32,3)
x_test = x_test.reshape(10000,32,32,3)

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', input_shape=(32,32,3)))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(20, (2,2), activation='relu')) 
model.add(MaxPooling2D()) 
model.add(Flatten()) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# #3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=100, batch_size=1500, callbacks=[es], validation_split=0.1, verbose=2)


#4. 평가, 예측
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('acc :', acc[-1])
print('val_acc: ', val_acc[-1])

loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# 증폭
# loss :  1.1680935621261597
# accuracy :  0.6295999884605408

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