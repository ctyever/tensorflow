from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping


(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

# print(x_train[0], type(x_train[0])) # class list # AttributeError: 'list' object has no attribute 'shape'
# print(y_train[0])
# print(np.unique(y_train))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45]

# print(len(x_train[0]), len(x_train[1])) # 87 56
# print(x_train)
# print(x_train.shape, x_test.shape) # (8982,) (2246,)
# print(y_train.shape, y_test.shape) # (8982,) (2246,)

# print(type(x_train)) # <class 'numpy.ndarray'>

# print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) # 뉴스기사의 최대길이 :  2376
# print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) # 뉴스기사의 최대길이 :  2376

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
# print(x_train.shape, x_test.shape) # (8982, 100) (2246, 100)
# print(type(x_train), type(x_train[0])) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
# print(x_train[1])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # (8982, 46) (2246, 46)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

model = Sequential()
                 # 단어사전의 개수                   단어수, 길이
model.add(Embedding(input_dim=10000, output_dim=11, input_length=100))
# input_length 안 쒀줘도 되는데 자동으로 인식하면서 None 으로 인식함
# model.add(Embedding(128, 77)) # input_dim 이 단어개수 보다 많으면 됨, 그런데 맞춰주는게 좋음
model.add(LSTM(32, activation='relu'))
model.add(Dense(46, activation='softmax'))

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=30, mode='min')

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es], validation_split=0.1)
end_time = time.time() - start_time

# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)


'''
1차
acc :  0.6460373997688293
'''
