from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np 
import pandas as pd
from tensorflow.python.keras.backend import conv1d

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
'''
[[13278100.    19100.    19320.    19000.    19160.] 
 [13724400.    19120.    19220.    18980.    19160.] 
 [16811200.    19100.    19100.    18840.    18840.] 
 ...
 [13155414.    79100.    79200.    78800.    79000.] 
 [12456646.    78500.    79000.    78400.    79000.] 
 [12355296.    79000.    79100.    78500.    78500.]]
'''
# print(np.min(datasets), np.max(datasets)) # 0.0 90306177.0

datasets2 = datasets2.to_numpy()
datasets2 = datasets2[999:-1, :]


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
# print(datasets.shape)  # (2597, 5, 5) 
# print(datasets)
# print(datasets[2596, 4, 4]) # 78500.0

x1 = datasets[:2594, :, :]
x2 = datasets2[:2594, :, :]
y = datasets[3:, 4, 1] #  [2:, 4, 4] 2틀 후 종가 / [3:, 4, 1] 3일 후 시가
# print(y.shape) #(2594,)
# print(y) # 이틀 후 종가 [18260. 18600. 18440. ... 79000. 79000. 78500.] / 3일후 시가 [18280. 18980. 18560. ... 79100. 78500. 79000.]
# print(x1.shape) # (2594, 5, 5)
# # print(x1[2594,:, :])

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, 
         train_size=0.8, shuffle=True, random_state=66)

x1_train = x1_train.reshape(x1_train.shape[0], 5*5)
x1_test = x1_test.reshape(x1_test.shape[0], 5*5)
x2_train = x2_train.reshape(x2_train.shape[0], 5*5)
x2_test = x2_test.reshape(x2_test.shape[0], 5*5)

x1_predict = datasets[2596, :, :]
x2_predict = datasets2[2596, :, :]
# print(x1_predict, x2_predict)
x1_predict = x1_predict.reshape(1, 25)
x2_predict = x2_predict.reshape(1, 25)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()
scaler = RobustScaler()
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

# # 2. 모델 구성
# input1 = Input(shape=(5, 5))
# conv1 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(input1)
# lstm1 = LSTM(units=128, activation='relu')(conv1)
# output1 = Dense(64, activation='relu')(lstm1)
# output2 = Dense(64, activation='relu')(output1)
# output3 = Dense(32, activation='relu')(output2)
# output4 = Dense(16, activation='relu' )(output3)
# output5 = Dense(8, activation='relu')(output4)
# output6 = Dense(3, activation='relu')(output5)


# input2 = Input(shape=(5, 5))
# conv2 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(input2)
# lstm2 = LSTM(units=128, activation='relu')(conv2)
# output7 = Dense(64, activation='relu')(lstm1)
# output8 = Dense(64, activation='relu' )(output7)
# output9 = Dense(32, activation='relu')(output8)
# output10 = Dense(16, activation='relu')(output9)
# output11 = Dense(8, activation='relu')(output10)
# output12 = Dense(4, activation='relu')(output11)

# from tensorflow.keras.layers import concatenate
# merge1 = concatenate([output6, output12])
# merge2 = Dense(3, activation='relu')(merge1)
# lastoutput = Dense(1, activation='relu')(merge1)

# model = Model(inputs= [input1, input2], outputs=lastoutput)

# # model.summary()

# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from keras.callbacks import EarlyStopping, ModelCheckpoint

# es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1,
#                     restore_best_weights=True)

# ##########################################################
# import datetime
# date = datetime.datetime.now()
# date_time = date.strftime("%m%d_%H%M")

# filepath = './_save/'
# filename = '.{epoch:04d}-{val_loss:.4f}.hdf5'
# modelpath = "".join([filepath, "samsung", date_time, "_", filename])
# #################################################################

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
#                        filepath= modelpath) 


# model.fit([x1_train, x2_train], y_train, epochs=1000, batch_size=16, callbacks=[es, mcp], validation_split=0.1, verbose=2)

# model.save('./_save/samsung.h5')

model = load_model('./_save/samsung_CTY2_79364.945.hdf5')
#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)

# print(x1_train.shape, x2_train.shape, y_train.shape) 

y_predict = model.predict([x1_predict, x2_predict])

print('예측값 : ', y_predict)


'''
2차 월요일 시가
loss :  3706938.5  / samsung0723_1003_.0124-2644930.7500
예측값 :  [[80041.51]]

loss :  3708229.5 / samsung0723_1046_.0193-2601396.0000
예측값 :  [[80290.16]]

scaler = RobustScaler()
loss :  3588108.5 / samsung0723_1735_.0092-2568633.7500
예측값 :  [[79962.734]]

loss :  3619940.75 / samsung0723_1743_.0103-2573891.2500
예측값 :  [[79651.44]]

모델 변경
loss :  3617712.75 / samsung0725_2025_.0071-2597637.2500
예측값 :  [[79787.914]]

모델변경
loss :  2754707.0 / samsung_CTY2_79364.945
예측값 :  [[79364.945]]


'''






