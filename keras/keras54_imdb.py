from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)

# print(x_train[0], type(x_train[0])) # <class 'list'>
# print(x_train.shape, x_test.shape) # (25000,) (25000,)
# print(y_train.shape, y_test.shape) # (25000,) (25000,)

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=500, padding='pre')
x_test = pad_sequences(x_test, maxlen=500, padding='pre')
# print(x_train.shape, x_test.shape)  # (25000, 500) (25000, 500)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# print(y_train.shape, y_test.shape) # (25000, 2) (25000, 2)
# print(np.unique(y_train)) # [0. 1.]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

model = Sequential()
                 # 단어사전의 개수                   단어수, 길이
model.add(Embedding(input_dim=10000, output_dim=11, input_length=500))

model.add(LSTM(32, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# model.summary()

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping

# es = EarlyStopping(monitor='val_loss', patience=30, mode='min')

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=1, batch_size=32)
end_time = time.time() - start_time

# 4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print("acc : ", acc)

# acc :  0.5