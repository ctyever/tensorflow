import numpy as np 
from sklearn.model_selection import train_test_split

x = np.load('./_save/_npy/k59_6_x.npy')
y = np.load('./_save/_npy/k59_6_y.npy')


x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66
)

# print(x_train.shape, x_test.shape) # (1764, 150, 150, 3) (756, 150, 150, 3)
# print(y_train.shape, y_test.shape) # (1764, 3) (756, 3)


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
model.add(Dense(3, activation='softmax'))

# model.summary()
# model.trainable=False # 전체 모델 훈련을 동결한다

# 3. 평가, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                         filepath='./_save/ModelCheckPoint/keras48_MCP_cifar10.hdf5')

model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es,], validation_split=0.08, verbose=2)


loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
xception / True / GAP
loss :  0.6823021769523621 
accuracy :  0.8492063283920288

xception
loss :  0.9191671013832092
accuracy :  0.8214285969734192

1차
acc :  0.3849206268787384

2차 클래스 분류하고 다시 돌림
acc :  0.591269850730896

'''