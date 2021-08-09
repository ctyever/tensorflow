import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn import datasets  
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (569, 30) (569,)

# print(np.unique(y)) # (0, 1)


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

# 2. 모델 구성
input1 = Input(shape=(30,1))
# lstm = LSTM(units=128, activation='relu')(input1)
conv1 = Conv1D(128, 2, activation='relu')(input1)
flatten1 = Flatten()(conv1)
dense1 = Dense(128, activation='relu', name='dense1')(flatten1)
dense2 = Dense(64, activation='relu', name='dense2')(dense1)
dense3 = Dense(64, activation='relu', name='dense3')(dense2)
dense4 = Dense(64, activation='relu', name='dense4')(dense3)
dense5 = Dense(32, activation='relu', name='dense5')(dense4)
output1 = Dense(1, activation='sigmoid', name='output1')(dense5)
#sigomid 이진분류에서 쓰는 활성화 함수, 마지막 layer에 100% sigmoid

model = Model(inputs= input1, outputs=output1)

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

optimizer = Adam(lr = 0.001)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) # 이진분류에서 loss 값은  binary_crossentropy  

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

model.fit(x_train, y_train, epochs=1000, batch_size=8, callbacks=[es, reduce_lr], validation_split=0.3, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
# y_predict = model.predict(x_test)

# learnig rate
# loss :  0.03554888814687729
# accuracy :  0.9824561476707458

# conv1d
# loss :  0.02890210598707199
# accuracy :  0.988304078578949

# lstm
# loss :  0.10537596046924591
# accuracy :  0.9590643048286438

# loss :  0.04805237427353859

# # 예측
# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)
