from icecream import ic
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
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
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)

# 2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim=10))
model.add(Dense(100))
model.add(Dense(3))
model.add(Dense(100))
model.add(Dense(3))
model.add(Dense(1))

# model.summary()


# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
                        filepath='./_save/ModelCheckPoint/keras47_MCP.hdf5')
import time
start_time = time.time()
model.fit(x_train, y_train, epochs=300, batch_size=32, callbacks=[es, cp], validation_split=0.08, verbose=2)
end_time = time.time() - start_time

model.save('./_save/ModelCheckPoint/keras47_model_save.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time )
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("r2스코어 : ", r2)

# print('예측값 : ', y_predict)

# load 한 모델로 실행
# r2스코어 :  0.6052078497306375
# early stop 적용
# 걸린 시간 :  3.6189491748809814
# loss :  2419.3681640625
# r2스코어 :  0.5622061602948372

# # epochs=1000, batch_size=1, validation_split=0.3, loss :  3075.999267578125, r2스코어 :  0.5062895077464991
# # 1000, 3, 0.3, loss :  3087.773193359375, r2스코어 :  0.5043997328171351
# # 1000, 1, 0.3, loss :  3090.016357421875 r2스코어 :  0.5040396986759945
# # 1000, 1, 0.3, layer node + 층 + , loss :  3150.476318359375, r2스코어 :  0.49433568914503734
# # 1000, 1, 0.3, 층 - ,  loss :  3094.7932128906255, r2스코어 :  0.5032729684162847
# # 1000, 1, validation 삭제, loss :  3064.811279296875 r2스코어 :  0.5080852062615169
# # 1000, 32, loss :  2997.41748046875, r2스코어 :  0.5189021670153582
# (x_train, y_train, epochs=300, batch_size=32, validation_split=0.08, verbose=2)  r2스코어 :  0.6125792352410648 / r2스코어 :  0.614444874784188 / r2스코어 :  0.6124839106429363

# #과제 1
# # 0.62 까지 올릴것



