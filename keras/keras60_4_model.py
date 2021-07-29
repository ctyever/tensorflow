from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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

# test_datagen = ImageDataGenerator( rescale=1./255 )

# 1. ImageDataGenerator 를 정의
# 2. 파일에서 땡겨올려면 -> flow_from_directory() // x, y가 튜플 형태로 뭉쳐있어
# 3. 데이터에서 땡겨올려면 -> flow()              // x, y가 나눠있어

augment_size = 40000

randix = np.random.randint(x_train.shape[0], size=augment_size)
# print(x_train.shape[0]) # 60000
# print(randix)           # [35050  6394 58848 ... 47817 59839 14861]
# print(randix.shape)     # (40000, )

x_augmented = x_train[randix].copy()
y_augmented = y_train[randix].copy()
# print(x_augmented.shape) #( 40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                batch_size=augment_size, shuffle=False).next()[0]

# print(x_augmented.shape) # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# print(x_train.shape, y_train.shape)  #(100000,28,28,1) (100000)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)

x_train = x_train.reshape(100000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(100000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

# 2. 모델 구성

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', input_shape=(28,28, 1)))
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

# 데이터 증폭
# loss :  0.720223605632782
# accuracy :  0.8996000289916992
# acc : 0.9130111336708069
# val_acc:  0.7581999897956848
# loss :  0.4601576626300812
# accuracy :  0.9038000106811523

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



