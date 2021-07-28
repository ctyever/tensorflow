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

# predict_datagen = ImageDataGenerator( rescale=1./255 )

xy = datagen.flow_from_directory(
    './data/rps/rps',
    target_size=(150, 150),
    batch_size=840,
    class_mode='categorical',
    classes=['paper,','rock','scissors'], 
)

# full = train_datagen.flow_from_directory(
#     '../data/rps',
#     target_size=(150,150),
#     batch_size=840,
#     class_mode='categorical',
#     classes=['paper,','rock','scissors'],
# )

# Found 2520 images belonging to 3 classes.

# pred = predict_datagen.flow_from_directory(
#     './data/cty',
#     target_size=(150, 150),
#     batch_size=1,
#     class_mode='categorical'   
# )


# print(xy[0][0])  # x 값
# print(xy[1][0])  # x 값
# print(xy[0][1])  # y 값
# print(xy.class_indices)
# print(pred[0][1])  # y 값
# print(pred[0][0].shape)
# print(xy[0][0].shape, xy[0][1].shape) # (840, 150, 150, 3) (840, 3)

np.save('./_save/_npy/k59_6_x.npy', arr=xy[0][0])
np.save('./_save/_npy/k59_6_y.npy', arr=xy[0][1])
# np.save('./_save/_npy/k59_5_pred.npy', arr=pred[0][0])



