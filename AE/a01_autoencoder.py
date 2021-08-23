# 앞뒤가 똑같은 오~토 인코더~~

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.engine.input_layer import Input

# 1. 데이터
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

input_img = Input(shape=(784,))
# encoded = Dense(64, activation='relu')(input_img)
encoded = Dense(1064, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='relu')(encoded) # 시그모이드 아닌건 범위가 넓어지다 보니 더 흐릿해짐
# decoded = Dense(784, activation='linear')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)

autoencoder = Model(input_img, decoded)

# autoencoder.summary()

# 3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='mse') # sigmoid 로 했을대 바이너리 써도 별 차이 없음


autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()