from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np 
import pandas as pd

# 1. 데이터 구성
datasets = pd.read_csv('./data/samsung.csv',  sep=',', 
                        index_col=None, header=0, encoding='EUC-KR')

datasets2 = pd.read_csv('./data/skhynix.csv',  sep=',', 
                        index_col=None, header=0, encoding='EUC-KR')
                        
# print(datasets)
# print(datasets.shape) # (3601, 16)

datasets = datasets.sort_values('일자')
datasets2 = datasets2.sort_values('일자')
# # print(datasets)
# # print(datasets.info())
# # print(datasets.describe()) # 시가, 고가, 저가, 종가, 거래량

datasets = datasets.loc[:, ['거래량', '시가', '고가', '저가', '종가']]
datasets2 = datasets2.loc[:, ['거래량', '시가', '고가', '저가', '종가']]
# print(datasets.shape) # (3601, 5)
# print(datasets2)

datasets = datasets.to_numpy()
datasets = datasets[999:-1, :]
# print(datasets)

datasets2 = datasets2.to_numpy()
datasets2 = datasets2[999:-1, :]

'''

'''

# print(np.min(datasets), np.max(datasets))

size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i+size, :]
        aaa.append(subset)
    return np.array(aaa)

datasets = split_x(datasets, size)

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i:i+size, :]
        aaa.append(subset)
    return np.array(aaa)

datasets2 = split_x(datasets2, size)
# print(datasets.shape)  # (2597, 5, 5) / (2582, 20, 5)
# print(datasets)
# print(datasets[2596, 4, 4]) # 78500.0

x1 = datasets[:2595, :, :]
x2 = datasets2[:2595, :, :]
y = datasets[2:, 4, 4]
# print(y.shape) #(2595,)
# print(y) # [18260. 18600. 18440. ... 79000. 79000. 78500.]
# print(x1.shape) # (2595, 5, 5)
# print(x1[2594,:, :])

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, 
         train_size=0.7, shuffle=False)

x1_train = x1_train.reshape(x1_train.shape[0], 5*5)
x1_test = x1_test.reshape(x1_test.shape[0], 5*5)
x2_train = x2_train.reshape(x2_train.shape[0], 5*5)
x2_test = x2_test.reshape(x2_test.shape[0], 5*5)


x1_predict = datasets[2596, :, :]
x2_predict = datasets2[2596, :, :]
x1_predict = x1_predict.reshape(1, 25)
x2_predict = x2_predict.reshape(1, 25)
# print(x1_predict, x2_predict) # (1, 25) (1, 25)
# x1_predict = scaler.inverse_transform(x1_predict)
# x2_predict = scaler.inverse_transform(x2_predict)

# print(x1_predict)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
x1_train = scaler.fit_transform(x1_train)
x1_test = scaler.transform(x1_test)
x1_predict = scaler.transform(x1_predict)
x2_train = scaler.fit_transform(x2_train)
x2_test = scaler.transform(x2_test)
x2_predict = scaler.transform(x2_predict)

x1_train = x1_train.reshape(x1_train.shape[0], 5, 5)
x1_test = x1_test.reshape(x1_test.shape[0], 5, 5)
x2_train = x2_train.reshape(x2_train.shape[0], 5, 5)
x2_test = x2_test.reshape(x2_test.shape[0], 5, 5)

x1_predict = x1_predict.reshape(1,5,5)
x2_predict = x2_predict.reshape(1,5,5)


# 2. 모델 구성
input1 = Input(shape=(5, 5))
lstm1 = LSTM(units=16, activation='relu')(input1)
output1 = Dense(3, name='output1')(lstm1)

input2 = Input(shape=(5, 5))
lstm2 = LSTM(units=16, activation='relu')(input2)
output2 = Dense(4, name='output2')(lstm2)

from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2])
merge2 = Dense(3, activation='relu')(merge1)
lastoutput = Dense(1, activation='relu')(merge1)

model = Model(inputs= [input1, input2], outputs=lastoutput)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1,
                    restore_best_weights=True)

##########################################################
import datetime
date = datetime.datetime.now()
date_time = date.strftime("%m%d_%H%M")

filepath = './_save/'
filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "samsung", date_time, "_", filename])
#################################################################

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                       filepath= modelpath) 


model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=16, callbacks=[es, mcp], validation_split=0.1, verbose=2)

# # model.save('./_save/samsung.h5')

# model = load_model('./_save/samsung_CTY1_78601.57.hdf5')
#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)

# print(x1_train.shape, x2_train.shape, y_train.shape) # (1816, 5, 5) (1816, 5, 5) (1816,)

y_predict = model.predict([x1_predict, x2_predict])

print('예측값 : ', y_predict)


# 5. r2 구하기
# r2 = r2_score(y_test, y_predict)

# print("r2스코어 : ", r2)

# epochs=100, batch_size=16
# loss :  7497013.5 / samsung0722_2051_.0029-5277378.0000
# 예측값 :  [[82554.5]]

# loss :  5976455.0 / samsung0722_2107_.0036-4111033.7500
# 예측값 :  [[78237.04]]

# epochs=1000, batch_size=16, patience=20 / samsung0722_2127_.0047-4156188.2500
# loss :  5843443.0
# 예측값 :  [[78619.29]]

# loss :  5828852.0  / samsung0722_2204_.0048-4166074.0000
# 예측값 :  [[78601.57]]

# loss :  5808767.5 / samsung0723_0729_.0325-4202796.0000
# 예측값 :  [[78399.83]]







