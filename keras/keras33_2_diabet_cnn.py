from icecream import ic
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
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

ic(x_train.shape, x_test.shape) # ic| x_train.shape: (309, 10), x_test.shape: (133, 10)


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

x_train = x_train.reshape(309, 5, 2, 1)
x_test = x_test.reshape(133, 5, 2, 1)

#2. 모델구성

# 모델 1
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(5, 2, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
# model.add(MaxPooling2D()) 이거 에러남, 왜??

# model.add(Conv2D(64, (2, 2), activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2, 2), padding='same', activation='relu')) 
# model.add(MaxPool2D()) 

# model.add(Flatten()) 
# model.add(Dense(128, activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='relu'))

# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.08, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
# print('예측값 : ', y_predict)

#5. r2 구하기
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)

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

# cnn 
# loss :  3182.7265625
# r2스코어 :  0.4240735361107344

# cnn, 모델 수정
# loss :  3176.20947265625
# r2스코어 :  0.4252528982346453