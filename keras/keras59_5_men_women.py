# 실습 1.
# men women 데이터로 모델링을 구성할 것!

# 실습 2. 
# 본인 사진으로 predict 하시오!!

import numpy as np 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
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

predict_datagen = ImageDataGenerator( rescale=1./255 )

xy = datagen.flow_from_directory(
    './data/men_women',
    target_size=(150, 150),
    batch_size=3400,
    class_mode='binary',
    classes=['men', 'women']  
)

# 그냥 실행했을 때 Found 3309 images belonging to 2 classes.

pred = predict_datagen.flow_from_directory(
    './data/cty',
    target_size=(150, 150),
    batch_size=1,      
)

# print(xy.class_indices) # {'men': 0, 'women': 1}
# print(xy[0][0])  # x 값
# print(xy[1][0])  # x 값
# print(xy[0][1])  # y 값
# print(pred[0][1])  # y 값
# print(pred[0][0].shape)
# print(xy[0][0].shape, xy[0][1].shape) # (3309, 150, 150, 3) (3309,)

np.save('./_save/_npy/k59_5_x.npy', arr=xy[0][0])
np.save('./_save/_npy/k59_5_y.npy', arr=xy[0][1])
np.save('./_save/_npy/k59_5_pred.npy', arr=pred[0][0])



