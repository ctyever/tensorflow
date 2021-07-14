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
xx = Dense(3)(input1)
xx = Dense(4)(xx)
xx = Dense(10)(xx)
output1 = Dense(3)(xx)  # 돌아 감, 단순한 시퀀셜일때, 

model = Model(inputs= input1, outputs=output1)
model.summary()