from icecream import ic
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn import datasets 
from sklearn.datasets import load_diabetes

# 1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# ic(x.shape, y.shape)  # x.shape: (442, 10), y.shape: (442,)
# ic(datasets.feature_names)  # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델구성

# 모델 1
input1 = Input(shape=(10, 1))
# lstm = LSTM(units=128, activation='relu')(input1)
conv1 = Conv1D(128, 2, activation='relu')(input1)
flatten1 = Flatten()(conv1)
dense1 = Dense(128, activation='relu', name='dense1')(flatten1)
dense2 = Dense(64, activation='relu', name='dense2')(dense1)
dense3 = Dense(64, activation='relu', name='dense3')(dense2)
dense4 = Dense(32, activation='relu', name='dense4')(dense3)
dense5 = Dense(32, activation='relu', name='dense5')(dense4)
dense6 = Dense(16, activation='relu', name='dense6')(dense5)
dense7 = Dense(16, activation='relu', name='dense7')(dense6)
dense8 = Dense(8, activation='relu', name='dense8')(dense7)
dense9 = Dense(8, activation='relu', name='dense9')(dense8)
output1 = Dense(1, name='output1')(dense9)

model = Model(inputs= input1, outputs=output1)

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.08, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
# print('예측값 : ', y_predict)

#5. r2 구하기
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)

# conv1d
# loss :  2293.232666015625
# r2스코어 :  0.5850309137131717

# MinMaxScaler
# loss :  4216.40283203125
# r2스코어 :  0.23702593753959855

# MinMaxScaler
# loss :  1990.69287109375
# r2스코어 :  0.6397765814915889

# StandardScaler
# loss :  2639.744873046875
# r2스코어 :  0.5223281616003655

# MaxAbsScaler
# loss :  2314.134033203125
# r2스코어 :  0.5812486788960357

# RobustScaler
# loss :  2646.519287109375
# r2스코어 :  0.5211022965013998

# QuantileTransformer
# loss :  2357.136962890625
# r2스코어 :  0.5734671046194194

# PowerTransformer
# loss :  3083.09814453125
# r2스코어 :  0.4421016966556396