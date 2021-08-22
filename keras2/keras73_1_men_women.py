# 가장 잘 나온 전이학습모델로
# 이 데이터를 학습시켜서 결과치 도출
# keras 59번과의 성능 비교

import numpy as np 
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Conv1D, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import Xception
import tensorflow as tf

import numpy as np 
from sklearn.model_selection import train_test_split

x = np.load('./_save/_npy/k59_5_x.npy')
y = np.load('./_save/_npy/k59_5_y.npy')
pred = np.load('./_save/_npy/k59_5_pred.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66
)


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape) 

x_train=tf.image.resize(x_train,[71,71])
x_test=tf.image.resize(x_test,[71,71])
pred=tf.image.resize(pred,[71,71])

# 2. 모델 구성
xception = Xception(weights='imagenet', include_top=False, input_shape=(71, 71, 3))

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
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                         filepath='./_save/ModelCheckPoint/keras48_MCP_cifar10.hdf5')

model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es,], validation_split=0.08, verbose=2)


loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

result = model.predict(pred)
pred = np.argmax(result, axis = 1)
print('예측값 : ', pred)

'''
xception / True / GAP
loss :  1.4124822616577148
accuracy :  0.6727089881896973
예측값 :  [0]

xception
loss :  1.4523738622665405
accuracy :  0.634441077709198
예측값 :  [0]

증폭
acc :  0.617321252822876

1차
acc :  0.6092648506164551
예측값 :  [0]

acc :  0.6304128766059875
예측값 :  [0]
'''