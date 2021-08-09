from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM, Conv1D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd

datasets = pd.read_csv('./data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)

# print(datasets)
# print(datasets.shape)  # (4898, 12)

datasets.values
datasets = datasets.to_numpy()
# print(datasets)

x = datasets[:, :11]
y = datasets[:, 11:]
# print(y)

# print(x.shape, y.shape) # (4898, 11) (4898, 1)
# print(np.unique(y)) # [3. 4. 5. 6. 7. 8. 9.]
from sklearn.preprocessing import LabelEncoder
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
input1 = Input(shape=(11,1))
# lstm = LSTM(units=128, activation='relu')(input1)
conv1 = Conv1D(128, 2, activation='relu')(input1)
flatten1 = Flatten()(conv1)
dense1 = Dense(128, activation='relu', name='dense1')(flatten1)
dense2 = Dense(64, activation='relu', name='dense2')(dense1)
dense3 = Dense(64, activation='relu', name='dense3')(dense2)
dense4 = Dense(64, activation='relu', name='dense4')(dense3)
dense5 = Dense(32, activation='relu', name='dense5')(dense4)
output1 = Dense(7, activation='softmax', name='output1')(dense5)

model = Model(inputs= input1, outputs=output1)

# print(model.summary())

# #3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam

optimizer = Adam(lr = 0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)

model.fit(x_train, y_train, epochs=100, batch_size=8, callbacks=[es, reduce_lr], validation_split=0.1, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# learning rate
# loss :  1.1168171167373657
# accuracy :  0.5489795804023743

#conv1d
# loss :  1.081992745399475
# accuracy :  0.5360544323921204

# lstm
# loss :  1.1413252353668213
# accuracy :  0.5034013390541077

# # # loss :  1.0831924676895142
# # # accuracy :  0.5401360392570496