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

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

# print(x.shape) # (506, 13)
# print(y.shape) # (506,)

# print(datasets.feature_names)
# print(datasets.DESCR) # feature 자세히 나옴
# print(datasets['filename'])

#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=8)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
print('예측값 : ', y_predict)

#5. r2 구하기
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)

# epochs=1000, batch_size=1, loss :  17.291133880615234 r2스코어 :  0.7907075553752195
# 1000, 1, layer 변경 , loss :  17.116796493530273, r2스코어 :  0.7928177634040774
# 10000, 1, loss :  16.508121490478516, r2스코어 :  0.8001851831896087
# 100, 8, layer 변경, loss :  40.854827880859375, r2스코어 :  0.5397263378159165






