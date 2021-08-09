import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import load_iris

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (150, 4) (150,)
# print(y) 
'''
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
 '''

# 원핫인코딩 One-Hot-Encoding  (150,) -> (150, 3)
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

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
# lstm = LSTM(units=128, activation='relu')(input1)
conv1 = Conv1D(128, 2, activation='relu')(input1)
flatten1 = Flatten()(conv1)
dense1 = Dense(128, activation='relu', name='dense1')(flatten1)
dense2 = Dense(64, activation='relu', name='dense2')(dense1)
dense3 = Dense(64, activation='relu', name='dense3')(dense2)
dense4 = Dense(64, activation='relu', name='dense4')(dense3)
dense5 = Dense(32, activation='relu', name='dense5')(dense4)
output1 = Dense(3, activation='softmax', name='output1')(dense5) # 다중분류에서는 softmax


model = Model(inputs= input1, outputs=output1)

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

optimizer = Adam(lr = 0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

model.fit(x_train, y_train, epochs=100, batch_size=8, callbacks=[es, reduce_lr], validation_split=0.3, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# learnig rate
# loss :  0.03891993686556816
# accuracy :  1.0

# conv1d
# loss :  0.026436353102326393
# accuracy :  1.0

# lstm
# loss :  0.1922423541545868
# accuracy :  0.9777777791023254

# 예측
# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)


