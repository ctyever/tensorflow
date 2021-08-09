# 훈련데이터를 10만개로 증폭할것!
# 완료 후 기존 모델과 비교
# save_dir 도 temp 에 넣을것

# 훈련데이터를 10만개로 증폭할것!
# 완료 후 기존 모델과 비교
# save_dir 도 temp 에 넣을것

import numpy as np 
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

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
# print(np.unique(y_train))

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
model.add(Dense(100, activation='softmax'))

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
# loss :  4.394509792327881
# accuracy :  0.24869999289512634

# checkpoint
# loss :  3.412015914916992
# acc :  0.23100000619888306

# load
# loss :  5.255951881408691
# acc :  0.21389999985694885

# conv1d
# loss :  5.591292381286621
# acc :  0.21369999647140503

# lstm
# loss :  nan
# acc :  0.009999999776482582

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

# dnn
# loss :  3.611995220184326
# acc :  0.19920000433921814