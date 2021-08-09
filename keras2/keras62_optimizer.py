import numpy as np 

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,5,6,4,7,8,9,11])


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adagrad, Adadelta, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.01) / keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# 0.1 / loss :  2103.659912109375 결과물 :  [[-1.1662221]]
# 0.01 / loss :  1.0449453592300415 결과물 :  [[12.295345]]
# 0.001 / loss :  0.8050398826599121 결과물 :  [[10.577949]]

# optimizer = Adagrad(lr=0.01) / keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
# 0.1 / loss :  4915.9375 결과물 :  [[-86.190994]]
# 0.01 / loss :  0.8172993659973145 결과물 :  [[11.86687]]
# 0.001 / loss :  0.6875752210617065 결과물 :  [[11.178724]]

# optimizer = Adadelta(lr=0.001) / keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# 0.1 / loss :  0.8496705889701843 결과물 :  [[10.466709]]
# 0.01 / loss :  0.6926072835922241 결과물 :  [[11.112927]]
# 0.001 / loss :  5.409200191497803 결과물 :  [[7.286256]]

# optimizer = Adamax(lr=0.001) / keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
# 0.1 / loss :  124.34605407714844 결과물 :  [[-5.8468604]]
# 0.01 / loss :  0.6891612410545349 결과물 :  [[11.292485]]
# 0.001 / loss :  0.6878948211669922 결과물 :  [[11.2498]]

# optimizer = RMSprop(lr=0.001) # keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
# 0.1 / loss :  12124104704.0 결과물 :  [[-121842.664]]
# 0.01 / loss :  12.303461074829102 결과물 :  [[4.3832116]]
# 0.001 / loss :  4.257794380187988 결과물 :  [[14.476814]]

optimizer = SGD() # keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# 0.1 / loss :  nan 결과물 :  [[nan]]
# 0.01 / loss :  nan 결과물 :  [[nan]] / 디폴트 loss :  nan 결과물 :  [[nan]]
# 0.001 / loss :  0.8098844289779663 결과물 :  [[10.530104]]


# optimizer = Nadam(lr=0.002) # keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
# 0.1 / loss :  56.86064529418945 결과물 :  [[22.217949]]
# 0.01 / loss :  2.7405219078063965 결과물 :  [[13.874267]]
# 0.001 / loss :  0.7062196731567383 결과물 :  [[10.955574]]
# 0.002 / loss :  1.043622374534607 결과물 :  [[10.111372]]


# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x,y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('loss : ', loss, '결과물 : ', y_pred)

