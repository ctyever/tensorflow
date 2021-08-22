# 실습
# cifar10 과 cifar 100 으로 모델 만들것
# trainable=True, False
# FC 로 만든것과 Avarage Pooling 으로 만든거 비교

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet101
import tensorflow as tf
from tensorflow.keras.datasets import cifar100, cifar10


# 1. 데이터 구성
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)
# 2차원으로 reshpae 하고 다시 4차원으로 원위치
# print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # 한번에 써줄 수 있음, train 에서만 쓴다
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], 32,32, 3)
x_test = x_test.reshape(x_test.shape[0], 32,32, 3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 2. 모델링
resnet101 = ResNet101(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# model = VGG16()
# model = VGG19()

resnet101.trainable=False

model = Sequential()
model.add(resnet101)
model.add(Flatten())
model.add(Dense(100, activation='relu'))
# model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

# model.summary()
# model.trainable=False # 전체 모델 훈련을 동결한다

# 3. 평가, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                         filepath='./_save/ModelCheckPoint/keras48_MCP_cifar10.hdf5')

model.fit(x_train, y_train, epochs=100, batch_size=1500, callbacks=[es,], validation_split=0.08, verbose=2)


loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])




'''
결과 출력
1. cifar 10
trainable = True, FC : loss=?, acc=?
loss :  5.049232006072998
accuracy :  0.1014999970793724
trainable = True, GAP : loss=?, acc=?
loss :  5.763908863067627
accuracy :  0.15780000388622284
trainable = False, FC : loss=?, acc=?
loss :  1.7060449123382568
accuracy :  0.3928999900817871
trainable = False, GAP : loss=?, acc=?
loss :  1.750329852104187
accuracy :  0.37610000371932983


2. cifar 100
trainable = True, FC : loss=?, acc=?
loss :  9.360396385192871
accuracy :  0.010099999606609344
trainable = True, GAP : loss=?, acc=?
loss :  1251251840.0
accuracy :  0.009999999776482582
trainable = False, FC : loss=?, acc=?
loss :  4.070448875427246
accuracy :  0.09099999815225601
trainable = False, GAP : loss=?, acc=?
loss :  4.186057090759277
accuracy :  0.08399999886751175
'''

