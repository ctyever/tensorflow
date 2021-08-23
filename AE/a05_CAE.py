# 2번 카피해서 복붙
# 딥하게 구성
# 2개의 모델을 구성하는데 하나는 기본적 오토인코더
# 다른 하나는 딥하게 구성
# 2개의 성능 비교
# 노드 개수 적게 딥하게 구성시 흐려짐, 노드개수 154로 유지시 비슷함

# 앞뒤가 똑같은 오~토 인코더~~
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.backend import sigmoid
from tensorflow.python.keras.engine.input_layer import Input

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255

# print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)


# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, UpSampling2D

def autoencoder1(hidden_layer_size):  # 기본적인 오토인코더
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), 
                    activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoencoder2(hidden_layer_size):  # 기본적인 오토인코더
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

# model1 = autoencoder1(hidden_layer_size=154)  # pca 95%

# model1.compile(optimizer='adam', loss='mse')

# model1.fit(x_train, x_train, epochs=10)

# output1 = model1.predict(x_test)

#### model2 는 딥하게 구성
model2 = autoencoder2(hidden_layer_size=154)

model2.summary()

model2.compile(optimizer='adam', loss='mse')

model2.fit(x_train, x_train, epochs=10)

output2 = model2.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다
# random_images = random.sample(range(output1.shape[0]), 5)

# # 원본(입력) 이미지를 맨 위에 그린다
# for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
#     ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
#     if i ==0:
#         ax.set_ylabel("INPUT", size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])

# # 오토인코더가 출력한 이미지를 아래에 그린다
# for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
#     ax.imshow(output1[random_images[i]].reshape(28, 28), cmap='gray')
#     if i ==0:
#         ax.set_ylabel("OUTPUT", size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])

# plt.tight_layout()
# plt.show()

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output2.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output2[random_images[i]].reshape(28, 28), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT2", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()



