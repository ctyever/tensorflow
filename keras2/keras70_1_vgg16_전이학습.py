from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
import os
import tensorflow as tf

model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
# model = VGG16()
# model = VGG19()

model.trainable=False

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))



# input_1 (InputLayer)         [(None, 32, 32, 3)]       0
# block1_conv1 (Conv2D)        (None, 32, 32, 64)        1792
# ...............................................................
# flatten (Flatten)            (None, 25088)             0
# _________________________________________________________________     
# fc1 (Dense)                  (None, 4096)              102764544      
# _________________________________________________________________     
# fc2 (Dense)                  (None, 4096)              16781312       
# _________________________________________________________________     
# predictions (Dense)          (None, 1000)              4097000        
# =================================================================     
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

# FC <- 용어정리
# 완전히 연결 되었다라는 뜻으로,
# 한층의 모든 뉴런이 다음층이 모든 뉴런과 연결된 상태로
# 2차원의 배열 형태 이미지를 1차원의 평탄화 작업을 통해 이미지를 분류하는데 사용되는 계층입니다.
# [출처] [딥러닝 레이어] FC(Fully Connected Layers)이란?|작성자 인텔리즈
