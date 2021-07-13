import numpy as np
from numpy.core.fromnumeric import transpose  

#1.데이터
x = np.array([range(100), range(301, 401), range(1, 101), range(100), range(401, 501)])

x = np.transpose(x)

print(x.shape)

y = np.array([range(711, 811), range(101,201)])
y = transpose(y)

# 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# 함수형과 Sequential 구조형은 성능 차이는 없지만 함수형은 재사용 할수 있다

input1 = Input(shape=(5,))
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(3)(dense3)

model = Model(inputs= input1, outputs=output1)
model.summary()


# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))

# model.summary()
# 
