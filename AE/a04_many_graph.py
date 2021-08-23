# 앞뒤가 똑같은 오~토 인코더~~

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.engine.input_layer import Input

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

# 2. 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), 
                    activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model01 = autoencoder(hidden_layer_size=1)
model02 = autoencoder(hidden_layer_size=2)
model03 = autoencoder(hidden_layer_size=4)
model04 = autoencoder(hidden_layer_size=8)
model05 = autoencoder(hidden_layer_size=16)
model06 = autoencoder(hidden_layer_size=32)

print("############# node 1개 시작 ###############")
model01.compile(optimizer='adam', loss='binary_crossentropy')
model01.fit(x_train, x_train, epochs=10)

print("############# node 2개 시작 ###############")
model02.compile(optimizer='adam', loss='binary_crossentropy')
model02.fit(x_train, x_train, epochs=10)

print("############# node 4개 시작 ###############")
model03.compile(optimizer='adam', loss='binary_crossentropy')
model03.fit(x_train, x_train, epochs=10)

print("############# node 8개 시작 ###############")
model04.compile(optimizer='adam', loss='binary_crossentropy')
model04.fit(x_train, x_train, epochs=10)

print("############# node 16개 시작 ###############")
model05.compile(optimizer='adam', loss='binary_crossentropy')
model05.fit(x_train, x_train, epochs=10)

print("############# node 32개 시작 ###############")
model06.compile(optimizer='adam', loss='binary_crossentropy')
model06.fit(x_train, x_train, epochs=10)

output01 = model01.predict(x_test)
output02 = model02.predict(x_test)
output03 = model03.predict(x_test)
output04 = model04.predict(x_test)
output05 = model05.predict(x_test)
output06 = model06.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, axes= plt.subplots(7, 5, figsize=(15, 15))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output01.shape[0]), 5)
outputs = [x_test, output01, output02, output03,
            output04, output05, output06]

# 원본(입력) 이미지를 맨 위에 그린다
for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28, 28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()



