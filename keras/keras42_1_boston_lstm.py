from icecream import ic
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
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
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. 모델구성

# 모델 1
input1 = Input(shape=(13, 1))
lstm = LSTM(units=128, activation='relu')(input1)
dense1 = Dense(128, activation='relu', name='dense1')(lstm)
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

model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.3, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
# print('예측값 : ', y_predict)

#5. r2 구하기
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)

# PowerTransformer, lstm
# loss :  18.230154037475586
# r2스코어 :  0.7946176815514603

# MinMaxScaler, batch_size=32
# loss :  11.4025239944458
# r2스코어 :  0.8715382774870997
# MinMaxScaler, batch_size=8, validation_split=0.08
# loss :  11.118603706359863
# r2스코어 :  0.8747369566931819
# MinMaxScaler, batch_size=8, validation_split=0.3, random_state=9
# loss :  9.450647354125977
# r2스코어 :  0.8935282796100319

# StandardScaler, batch_size=8, validation_split=0.3, random_state=9
# loss :  10.733266830444336
# r2스코어 :  0.8790781897189563

# MaxAbsScaler
# loss :  14.042271614074707
# r2스코어 :  0.8417986934838104

# RobustScaler
# loss :  9.446349143981934
# r2스코어 :  0.8935767002821393

# QuantileTransformer
# loss :  14.724702835083008
# r2스코어 :  0.8341103804678933

# PowerTransformer
# loss :  11.970605850219727
# r2스코어 :  0.8651382309639116

