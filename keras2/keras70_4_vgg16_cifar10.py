from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# 1. 데이터 구성
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델링
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# model = VGG16()
# model = VGG19()

vgg16.trainable=True # vgg 훈련을 동결한다

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

# model.trainable=False # 전체 모델 훈련을 동결한다

# 3. 평가, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                         filepath='./_save/ModelCheckPoint/keras48_MCP_cifar10.hdf5')

model.fit(x_train, y_train, epochs=100, batch_size=256, callbacks=[es,], validation_split=0.08, verbose=2)


loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# GAP
# loss :  0.7526970505714417
# accuracy :  0.7889999747276306

# vgg16.trainable=True 
# loss :  0.7394670248031616
# accuracy :  0.7888000011444092

# vgg16.trainable=False # vgg 훈련을 동결한다
# loss :  1.3481626510620117
# accuracy :  0.5942000150680542



