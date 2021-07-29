from tensorflow.keras.datasets import fashion_mnist
import numpy as np


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.10,
    shear_range=0.5,
    fill_mode='nearest'
)

# test_datagen = ImageDataGenerator( rescale=1./255 )

# xy_train = train_datagen.flow_from_directory(
#     './data/brain/train',
#     target_size=(150, 150),
#     batch_size=5,
#     class_mode='binary',
#     shuffle=True    
# )

# 1. ImageDataGenerator 를 정의
# 2. 파일에서 땡겨올려면 -> flow_from_directory() // x, y가 튜플 형태로 뭉쳐있어
# 3. 데이터에서 땡겨올려면 -> flow()              // x, y가 나눠있어
# print(x_train.shape) # (60000, 28, 28)
augument_size = 50
x_data = train_datagen.flow(
                np.tile(x_train[2].reshape(28*28), augument_size).reshape(-1, 28, 28, 1),    # x
                np.zeros(augument_size),        # y
                batch_size=augument_size,
                shuffle=False
).next()                     # iterator 방식으로 반환!!!

# print(type(x_data)) # <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
#                      # next() 입력하면서 -> <class 'tuple'>
# print(type(x_data[0])) # <class 'tuple'> -> <class 'numpy.ndarray'>
# # print(x_data[0][0]) # <class 'numpy.ndarray'>
# print(x_data[0][0].shape) # (28, 28, 1)  -> x 값
# print(x_data[0].shape) # (50, 28, 28, 1)
# print(x_data[1].shape) # (50,) -> y 값

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()







