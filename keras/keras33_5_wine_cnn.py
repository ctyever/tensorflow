from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd

datasets = pd.read_csv('./data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)

# print(datasets)
# print(datasets.shape)  # (4898, 12)

# print(datasets.info())
# print(datasets.describe())
# 1. 판다스 -> 넘파이
# x와 y를 분리
# sklear의 onehot??? 사용할것
# y의 라벨을 확인 np.unique(y)

# datasets.values
datasets = datasets.to_numpy()
# print(datasets)

x = datasets[:, :11]
y = datasets[:, 11:]
# print(y)

# print(x.shape, y.shape) # (4898, 11) (4898, 1)
# print(np.unique(y)) # [3. 4. 5. 6. 7. 8. 9.]
from sklearn.preprocessing import OneHotEncoder
oneHot_encoder = OneHotEncoder(sparse=False)  # 주의 : sparse=False
y = oneHot_encoder.fit_transform(y)
# print(np.unique(y))  # [0. 1.]


# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)


# print(y[:5])
# print(y.shape) # (4898, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

# print(y_test)
# print(y_train)
print(x_train.shape, x_test.shape) # (3428, 11) (1470, 11)


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

x_train = x_train.reshape(3428, 11, 1, 1)
x_test = x_test.reshape(1470, 11, 1, 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='same', activation='relu', input_shape=(13, 1, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(16, (2, 2), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(7, activation='softmax'))

'''
input1 = Input(shape=(11,))
dense1 = Dense(128, activation='relu', name='dense1')(input1)
dense2 = Dense(64, activation='relu', name='dense2')(dense1)
dense3 = Dense(64, activation='relu', name='dense3')(dense2)
dense4 = Dense(64, activation='relu', name='dense4')(dense3)
dense5 = Dense(32, activation='relu', name='dense5')(dense4)
output1 = Dense(7, activation='softmax', name='output1')(dense5)

model = Model(inputs= input1, outputs=output1)
'''

# print(model.summary())

# #3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=128, callbacks=[es], validation_split=0.1, verbose=2)
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time )
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# loss :  1.0831924676895142
# accuracy :  0.5401360392570496

# cnn
# 걸린 시간 :  6.795181512832642
# loss :  1.1935038566589355
# accuracy :  0.46870747208595276