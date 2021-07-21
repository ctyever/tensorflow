from icecream import ic
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn import datasets
from sklearn.datasets import load_boston

# 1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# ic(x.shape, y.shape)  # x.shape: (506, 13), y.shape: (506,)
# ic(datasets.feature_names) # datasets.feature_names: array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')
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
# input1 = Input(shape=(13, 1))
# # lstm = LSTM(units=128, activation='relu')(input1)
# conv1 = Conv1D(128, 2, activation='relu')(input1)
# flatten1 = Flatten()(conv1)
# dense1 = Dense(128, activation='relu', name='dense1')(flatten1)
# dense2 = Dense(64, activation='relu', name='dense2')(dense1)
# dense3 = Dense(64, activation='relu', name='dense3')(dense2)
# dense4 = Dense(32, activation='relu', name='dense4')(dense3)
# dense5 = Dense(32, activation='relu', name='dense5')(dense4)
# dense6 = Dense(16, activation='relu', name='dense6')(dense5)
# dense7 = Dense(16, activation='relu', name='dense7')(dense6)
# dense8 = Dense(8, activation='relu', name='dense8')(dense7)
# dense9 = Dense(8, activation='relu', name='dense9')(dense8)
# output1 = Dense(1, name='output1')(dense9)

# model = Model(inputs= input1, outputs=output1)

# model.summary()


#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
# # cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
# #                         filepath='./_save/ModelCheckPoint/keras48_MCP_boston.hdf5')

# # import time
# # start_time = time.time()
# model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.3, callbacks=[es, cp], verbose=2)
# # end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_model_boston_save.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_model_boston_save.h5') # save_model
model = load_model('./_save/ModelCheckPoint/keras48_MCP_boston.hdf5') #체크포인트

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
# print('예측값 : ', y_predict)

#5. r2 구하기
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)

# 체크포인트
# loss :  10.0862398147583
# r2스코어 :  0.8863676436212908

# load
# loss :  12.898103713989258
# r2스코어 :  0.8546889834223441

# conv1d, MinMaxScaler
# loss :  10.612846374511719
# r2스코어 :  0.8804348634967774






