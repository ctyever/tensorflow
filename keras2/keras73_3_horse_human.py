import numpy as np 
from sklearn.model_selection import train_test_split

x = np.load('./_save/_npy/k59_7_x.npy')
y = np.load('./_save/_npy/k59_7_y.npy')

# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, shuffle=True, random_state=66
)

# print(x_train.shape, x_test.shape) # (371, 150, 150, 3) (159, 150, 150, 3)
# print(y_train.shape, y_test.shape) # (371, 2) (159, 2)


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
loss :  0.4586102068424225 
accuracy :  0.9119496941566467

xception
loss :  0.6309812664985657
accuracy :  0.8742138147354126

1차
acc :  0.7106918096542358

'''