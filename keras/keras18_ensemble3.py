# 데이터
import numpy as np
from sklearn.model_selection import train_test_split

x1 = np.array([range(100), range(301, 401), range(1, 101)])
# x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x1 = np.transpose(x1)
# x2 = np.transpose(x2)
# y = np.array([range(1001, 1101)])
# y = np.transpose(y1)
y1 = np.array(range(1001, 1101))
y2 = np.array(range(1901, 2001))



print(x1.shape)

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2,
        train_size=0.7, shuffle=True, random_state=9)

print(x1_train.shape)


# 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

# 모델1
input1 = Input(shape=(3, ))
dense1 = Dense(100, activation='relu', name='dense1')(input1)
dense2 = Dense(70, activation='relu', name='dense2')(dense1)
dense3 = Dense(50, activation='relu', name='dense3')(dense2)
dense4 = Dense(30, activation='relu', name='dense4')(dense3)
dense5 = Dense(10, activation='relu', name='dense5')(dense4)
dense6 = Dense(5, activation='relu', name='dense6')(dense5)
output1 = Dense(1, name='output1')(dense6) 

# # 모델2
# input2 = Input(shape=(3, ))
# dense11 = Dense(10, activation='relu', name='dense11')(input2)
# dense12 = Dense(10, activation='relu', name='dense12')(dense11)
# dense13 = Dense(10, activation='relu', name='dense13')(dense12)
# dense14 = Dense(10, activation='relu', name='dense14')(dense13)
# output2 = Dense(4, name='output2')(dense14)

# from tensorflow.keras.layers import concatenate, Concatenate

# merge1 = concatenate([output1, output2])
# merge2 = Dense(10)(merge1)
lastoutput1 = Dense(1, name='lastoutput1')(output1)
lastoutput2 = Dense(1, name='lastoutput2')(output1)

model = Model(inputs = input1, outputs =[lastoutput1, lastoutput2])

model.summary()

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=8, verbose=1, validation_split=0.1)

#4. 평가 예측
loss = model.evaluate(x1_test, [y1_test, y2_test])
print('loss : ', loss)
# y_predict = model.predict([x1])
# print('예측값 : ', y_predict)
