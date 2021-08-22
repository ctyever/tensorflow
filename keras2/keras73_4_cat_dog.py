import numpy as np 


# 1. 데이터 구성

x_train = np.load('./_save/_npy/k59_8_train_x.npy')
y_train = np.load('./_save/_npy/k59_8_train_y.npy')
x_test = np.load('./_save/_npy/k59_8_test_x.npy')
y_test = np.load('./_save/_npy/k59_8_test_y.npy')

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception

xception = Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# model = VGG16()
# model = VGG19()

xception.trainable=True

model = Sequential()
model.add(xception)
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(2, activation='softmax'))

# model.summary()
# model.trainable=False # 전체 모델 훈련을 동결한다

# 3. 평가, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                         filepath='./_save/ModelCheckPoint/keras48_MCP_cifar10.hdf5')

model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es,], validation_split=0.08, verbose=2)


loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
xception / True / GAP
loss :  0.3892403841018677 
accuracy :  0.9110232591629028

loss :  0.27921262383461
accuracy :  0.9520514011383057

acc :  0.49975284934043884
'''


