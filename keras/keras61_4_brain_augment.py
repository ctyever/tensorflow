# 훈련데이터를 기존데이터 20% 더 할것
# 성과비교
# 증폭데이터는 temp에 저장 후 훈련 끝난 후 결과 본뒤 삭제

import numpy as np 
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D


x_train = np.load('./_save/_npy/k59_3_train_x.npy')
y_train = np.load('./_save/_npy/k59_3_train_y.npy')
x_test = np.load('./_save/_npy/k59_3_test_x.npy')
y_test = np.load('./_save/_npy/k59_3_test_y.npy')

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
# print(x_train.shape, x_test.shape) # (160, 150, 150, 3) (120, 150, 150, 3)
# print(np.unique(y_train))

augment_size = 300

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

# print(x_train.shape, y_train.shape)  # (460, 150, 150, 3) (460,)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) 

x_train = x_train.reshape(460, 150 * 150 * 3)
x_test = x_test.reshape(120, 150 * 150 * 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(460,150,150,3)
x_test = x_test.reshape(120,150,150,3)

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', input_shape=(150,150,3)))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(20, (2,2), activation='relu')) 
model.add(MaxPooling2D()) 
model.add(Flatten()) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

# #3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es], validation_split=0.1, verbose=2)


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
# loss :  0.8937498331069946
# accuracy :  0.5249999761581421