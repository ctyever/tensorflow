# 실습
# 앞뒤가 똑같은 오~토 인코더~~

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.engine.input_layer import Input

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train_noised = x_train + np.random.normal(0, 0.4, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.4, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D

def autoencoder(hidden_layer_size):  # 기본적인 오토인코더
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(3, 3), input_shape=(28,28, 1), padding='same',
                    activation='relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(64, (1, 1), padding='same'))
    model.add(MaxPool2D())
    model.add(Conv2D(64, (1, 1), padding='same'))
    # model.add(MaxPool2D())

    # model.add(Conv2D(32, (1, 1), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(32, (1, 1), padding='same'))
    model.add(UpSampling2D())
    model.add(Conv2D(32, (1, 1), padding='same'))
    # model.add(UpSampling2D())
    model.add(Conv2D(1, (1, 1), padding='same', activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154)  # pca 95% 154

model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs=10)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax11, ax12, ax13, ax14, ax15), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(3, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISED", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()




