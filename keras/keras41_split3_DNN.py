# 실습
# 1 - 100 까지의 데이터를

#      x               y  
# 1, 2, 3, 4, 5        6
# ...
# 95, 96, 97, 98, 99  100
import numpy as np  
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터 구성
x_data = np.array(range(1, 101))

y = np.array(range(6, 101))

x_predict = np.array(range(96, 106))

size = 6

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(x_data, size)
dataset2 = split_x(x_predict, 5)


# print(dataset2)
'''
[[ 96  97  98  99 100] 
 [ 97  98  99 100 101] 
 [ 98  99 100 101 102] 
 [ 99 100 101 102 103] 
 [100 101 102 103 104] 
 [101 102 103 104 105]]
'''

# print(dataset) 

'''
[[  1   2   3   4   5   6]
 [  2   3   4   5   6   7]
 [  3   4   5   6   7   8]
 ...    
 [ 94  95  96  97  98  99]
 [ 95  96  97  98  99 100]

'''
x = dataset[:, :5]
y = dataset[:, 5]
# print(x.shape, y.shape) (95, 5) (95,)


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
dataset2 = scaler.transform(dataset2)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)



#2. 모델구성
input1 = Input(shape=(5, 1))
# lstm = LSTM(units=128, activation='relu')(input1)
dense1 = Dense(128, activation='relu', name='dense1')(input1)
flatten1 = Flatten()(dense1)
dense2 = Dense(64, activation='relu', name='dense2')(flatten1)
dense3 = Dense(64, activation='relu', name='dense3')(dense2)
dense4 = Dense(32, activation='relu', name='dense4')(dense3)
dense5 = Dense(32, activation='relu', name='dense5')(dense4)
dense6 = Dense(16, activation='relu', name='dense6')(dense5)
dense7 = Dense(16, activation='relu', name='dense7')(dense6)
dense8 = Dense(8, activation='relu', name='dense8')(dense7)
dense9 = Dense(8, activation='relu', name='dense9')(dense8)
output1 = Dense(1, name='output1')(dense9)

model = Model(inputs= input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.3, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# dataset2 = dataset2.reshape(1, dataset2.shape[1], 1)
# print(dataset2.shape) # (6, 5, 1)
y_predict = model.predict(x_test)
# print(y_predict)
# print(y_test.shape)

# # 5. r2 구하기
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)

# DNN
# loss :  0.003230184316635132
# r2스코어 :  0.9999942759810397

# loss :  0.6141353249549866
# r2스코어 :  0.9989117270180008