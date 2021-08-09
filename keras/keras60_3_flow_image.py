from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from tensorflow.python.keras.backend import zeros 

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

# 1. ImageDataGenerator 를 정의
# 2. 파일에서 땡겨올려면 -> flow_from_directory() // x, y가 튜플 형태로 뭉쳐있어
# 3. 데이터에서 땡겨올려면 -> flow()              // x, y가 나눠있어

augment_size = 40000

randix = np.random.randint(x_train.shape[0], size=augment_size)
# print(x_train.shape[0]) # 60000
# print(randix)           # [35050  6394 58848 ... 47817 59839 14861]
# print(randix.shape)     # (40000, )

x_augmented = x_train[randix].copy()
y_augmented = y_train[randix].copy()
# print(x_augmented.shape) #( 40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                batch_size=augment_size, shuffle=False).next()[0]

print(x_augmented.shape) # (40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)  #(100000,28,28,1) (100000)

# 실습 1. x_augmented 10개와 원래 x_train 10개를 비교하는 이미지를 출력할 것!
# subplot(2, 10, ?) 사용


# import matplotlib.pyplot as plt
# plt.figure(figsize=(2,10))
# for i in range(20):    
#     if i <= 9:        
#         plt.subplot(2, 10, i+1)        
#         plt.axis('off')        
#         plt.imshow(x_train[i], cmap='gray')    
#     else:       
#         plt.subplot(2, 10, i+1)        
#         plt.axis('off')        
#         plt.imshow(x_train[i+60000], cmap='gray')

# plt.show()





