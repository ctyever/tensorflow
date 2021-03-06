# 과제3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn import datasets
from sklearn.datasets import load_boston

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# print(np.min(x), np.max(x)) # 0.0 711.0

#데이터 전처리 (0~1사이로 바꿈) 노말라이제이션, 정규화
# x = x/711.
# x = x/np.max(x)
# x = (x - np.min(x)) / (np.max(x) - np.min(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66)

        
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x.shape) # (506, 13)
# print(y.shape) # (506,)

# print(datasets.feature_names)
# print(datasets.DESCR) # feature 자세히 나옴
# print(datasets['filename'])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(100))
model.add(Dense(3))
model.add(Dense(100))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)


hist = model.fit(x_train, y_train, epochs=100, batch_size=8, callbacks=[es], validation_split=0.2)


print(hist.history.keys())
print(hist.history['loss'])
print(hist.history['val_loss'])

import matplotlib.pyplot as plt

import matplotlib
matplotlib.font_manager._rebuild()

plt.rc('font', family='NanumGothic')
print(plt.rcParams['font.family'])


plt.plot(hist.history['loss'] ) # x: epoch / y: hist.history['loss']
plt.plot(hist.history['val_loss'] )

plt.title("산점도")
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend(['train loss', 'val loss']) # 범례
plt.show()

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)
# y_predict = model.predict(x_test)
# # # print('예측값 : ', y_predict)

# #5. r2 구하기
# r2 = r2_score(y_test, y_predict)

# print("r2스코어 : ", r2)

# # epochs=1000, batch_size=1, loss :  17.291133880615234 r2스코어 :  0.7907075553752195
# # 1000, 1, layer 변경 , loss :  17.116796493530273, r2스코어 :  0.7928177634040774
# # 10000, 1, loss :  16.508121490478516, r2스코어 :  0.8001851831896087