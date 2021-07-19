import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn import datasets  
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(y[:100])

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
# print(x_train.shape, x_test.shape) # (398, 30) (171, 30)

x_train = x_train.reshape(398, 5, 6, 1)
x_test = x_test.reshape(171, 5, 6, 1)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(5, 2, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))
#sigomid 이진분류에서 쓰는 활성화 함수, 마지막 layer에 100% sigmoid


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 이진분류에서 loss 값은  binary_crossentropy  

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=8, callbacks=[es], validation_split=0.3, verbose=2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
# y_predict = model.predict(x_test)

# loss :  0.04805237427353859

# cnn
# loss :  0.09262897819280624
# accuracy :  0.9649122953414917

# 예측
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)
