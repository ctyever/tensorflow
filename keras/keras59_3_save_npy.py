import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    './data/brain/train',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary',
    shuffle=True    
)

# 그냥 실행했을 때 Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    './data/brain/test',
    target_size=(150, 150),
    batch_size=200,
    class_mode='binary'
)

# Found 120 images belonging to 2 classes.

# print(xy_train[0][0])  # x 값
# print(xy_train[0][1])  # y 값
# # print(xy_train[0][2]) # 없어
# print(xy_train[0][0].shape, xy_train[0][1].shape) # (160, 150, 150, 3) (160,)

np.save('./_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
np.save('./_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
np.save('./_save/_npy/k59_3_test_x.npy', arr=xy_test[0][0])
np.save('./_save/_npy/k59_3_test_y.npy', arr=xy_test[0][1])




