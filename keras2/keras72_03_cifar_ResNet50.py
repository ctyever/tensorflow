# 실습
# cifar10 과 cifar 100 으로 모델 만들것
# trainable=True, False
# FC 로 만든것과 Avarage Pooling 으로 만든거 비교

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import Xception
import tensorflow as tf
from tensorflow.keras.datasets import cifar100, cifar10
import tensorflow as tf


# 1. 데이터 구성
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# x_train = x_train.reshape(50000, 32 * 32 * 3)
# x_test = x_test.reshape(10000, 32 * 32 * 3)
# # 2차원으로 reshpae 하고 다시 4차원으로 원위치
# # print(x_train.shape, x_test.shape) # (50000, 3072) (10000, 3072)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train) # 한번에 써줄 수 있음, train 에서만 쓴다
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(x_train.shape[0], 32,32, 3)
# x_test = x_test.reshape(x_test.shape[0], 32,32, 3)

x_train=tf.image.resize(x_train,[71,71])
x_test=tf.image.resize(x_test,[71,71])

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 2. 모델링
xception = Xception(weights='imagenet', include_top=False, input_shape=(71, 71, 3))

# model = VGG16()
# model = VGG19()

xception.trainable=True

model = Sequential()
model.add(xception)
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

model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es,], validation_split=0.08, verbose=2)


loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])



'''
결과 출력
1. cifar 10
trainable = True, FC : loss=?, acc=?
loss :  0.5047889351844788
accuracy :  0.878600001335144
trainable = True, GAP : loss=?, acc=?
loss :  0.5193849802017212
accuracy :  0.8883000016212463
trainable = False, FC : loss=?, acc=?
loss :  1.9679077863693237
accuracy :  0.21809999644756317
trainable = False, GAP : loss=?, acc=?
loss :  2.1879396438598633
accuracy :  0.3815999925136566

2. cifar 100
trainable = True, FC : loss=?, acc=?
loss :  1.8298718929290771
accuracy :  0.6182000041007996
trainable = True, GAP : loss=?, acc=?
loss :  11.790122985839844
accuracy :  0.5917999744415283
trainable = False, FC : loss=?, acc=?

trainable = False, GAP : loss=?, acc=?

'''

