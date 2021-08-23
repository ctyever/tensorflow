# keras61_5 남자 여자 데이터에 노이즈를 넣어서
# 기미 주근께 여드름 제거하시오!

import numpy as np 
from sklearn.model_selection import train_test_split

x = np.load('./_save/_npy/k59_5_x.npy')
y = np.load('./_save/_npy/k59_5_y.npy')
pred = np.load('./_save/_npy/k59_5_pred.npy')

# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66
)

x_train = x_train.reshape(2316, 150, 150, 3) #.astype('float')/255
x_test = x_test.reshape(993, 150, 150, 3) #.astype('float')/255

x_train_noised = x_train + np.random.normal(0, 0.4, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.4, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# print(x_train.shape, x_test.shape) # (2316, 150, 150, 3) (993, 150, 150, 3)
# print(y_train.shape, y_test.shape) # (2316,) (993,)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, UpSampling2D

def autoEncoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size, kernel_size=(2, 2), 
                input_shape=(150, 150, 3),
                activation='relu', padding='same'))
    model.add(MaxPool2D(1,1))
    model.add(Conv2D(100, (2, 2), activation='relu', padding='same'))
    
    model.add(UpSampling2D(size=(1,1)))

    model.add(Conv2D(3, (2, 2), activation='sigmoid', padding='same'))
    return model

model = autoEncoder(hidden_layer_size=154)

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', verbose=1)

import time 

start_time = time.time()
hist = model.fit(x_train_noised, x_train, epochs=100, verbose=2,
    validation_split=0.05, callbacks=[es])
end_time = time.time() - start_time

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(3, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(150, 150, 3), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(150, 150, 3), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISED", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(150, 150, 3), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()