import numpy as np  
from numpy import array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM, GRU

# 1. 데이터
x1 = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
            [5,6,7], [6,7,8], [7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]]
            )
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])


x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

# 2. 모델 구성
# 모델1
input1 = Input(shape=(3,1))
gru1 = GRU(units=32, activation='relu')(input1)
dense1 = Dense(16, activation='relu')(gru1)
dense2 = Dense(16, activation='relu')(dense1)
dense3 = Dense(8, activation='relu')(dense2)
dense4 = Dense(8, activation='relu')(dense3)
output1 = Dense(4)(dense4)

# 모델2
input2 = Input(shape=(3,1))
gru2 = GRU(units=32, activation='relu')(input2)
dense11 = Dense(10, activation='relu', name='dense11')(gru2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13)
output2 = Dense(4, name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output2])
merge2 = Dense(10)(merge1)
lastoutput = Dense(1)(merge2)

model = Model(inputs= [input1, input2], outputs=lastoutput)

# model.summary()

# total params  = (INPUT + BIAS + OUTPUT ) * OUTPUT
# LSTM total params = ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1 * unit 개수) * 4

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)

model.fit([x1, x2], y, epochs=300, batch_size=1,  callbacks=[es])

# 4. 평가, 예측
# x_input = np.array([5,6,7]).reshape(1, 3, 1)

loss = model.evaluate([x1, x2], y)
print('loss : ', loss)
x1_predict = array([55,65,75])
x2_predict = array([65,75,85])
x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)

result = model.predict([x1_predict, x2_predict])
print(result)

# simple
# [[45.65882]]

#lstm
# [[85.71836]]

#gru
# loss :  [0.10304507613182068, 0.0]
# [[84.30732]]
