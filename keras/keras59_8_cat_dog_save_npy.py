# 실습
# categorical_crossentropy 와 sigmoid 조합


import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. 데이터 구성
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator( rescale=1./255 )

xy_train = train_datagen.flow_from_directory(
    './data/cat_dog/train',
    target_size=(150, 150),
    batch_size=8010,
    class_mode='binary',
    classes=['cats', 'dogs'] 
)

# Found 8005 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    './data/cat_dog/test',
    target_size=(150, 150),
    batch_size=2030,
    class_mode='binary',
    classes=['cats', 'dogs']
)

# Found 2023 images belonging to 2 classes.

# print(xy_train[0][0])  # x 값
# print(xy_train[0][1])  # y 값
# print(xy_train[0][2]) # 없어
print(xy_train[0][0].shape, xy_train[0][1].shape) # (8005, 150, 150, 3) (8005,)

np.save('./_save/_npy/k59_8_train_x.npy', arr=xy_train[0][0])
np.save('./_save/_npy/k59_8_train_y.npy', arr=xy_train[0][1])
np.save('./_save/_npy/k59_8_test_x.npy', arr=xy_test[0][0])
np.save('./_save/_npy/k59_8_test_y.npy', arr=xy_test[0][1])