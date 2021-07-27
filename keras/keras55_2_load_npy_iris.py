import numpy as np  
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

# 1. 데이터 구성 

x = np.load('./_save/_npy/k55_x_data_iris.npy')
y = np.load('./_save/_npy/k55_y_data_iris.npy')

# print(x_data)
# print(y_data)
# print(x_data.shape, y_data.shape) # (150, 4) (150,)

from tensorflow.keras.utils import to_categorical  #원핫인코딩
y = to_categorical(y)
# print(y[:5])
# print(y.shape) # (150, 3)

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
input1 = Input(shape=(4, 1))
lstm = LSTM(units=128, activation='relu')(input1)
dense1 = Dense(128, activation='relu', name='dense1')(lstm)
dense2 = Dense(64, activation='relu', name='dense2')(dense1)
dense3 = Dense(64, activation='relu', name='dense3')(dense2)
dense4 = Dense(64, activation='relu', name='dense4')(dense3)
dense5 = Dense(32, activation='relu', name='dense5')(dense4)
output1 = Dense(3, activation='softmax', name='output1')(dense5) # 다중분류에서는 softmax


model = Model(inputs= input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=8, callbacks=[es], validation_split=0.3, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# load_npy
# loss :  26.104782104492188
# r2스코어 :  0.7059015069563245

# load_npy
# loss :  0.06148451566696167
# accuracy :  1.0

# lstm
# loss :  0.1922423541545868
# accuracy :  0.9777777791023254

# 예측
# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)



