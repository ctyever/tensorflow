import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
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

# from tensorflow.keras.utils import to_categorical  #원핫인코딩
# y = to_categorical(y)
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

# 2. 모델 구성
# input1 = Input(shape=(4,))
# dense1 = Dense(128, activation='relu', name='dense1')(input1)
# dense2 = Dense(64, activation='relu', name='dense2')(dense1)
# dense3 = Dense(64, activation='relu', name='dense3')(dense2)
# dense4 = Dense(64, activation='relu', name='dense4')(dense3)
# dense5 = Dense(32, activation='relu', name='dense5')(dense4)
# output1 = Dense(3, activation='softmax', name='output1')(dense5) # 다중분류에서는 softmax


# model = Model(inputs= input1, outputs=output1)
from sklearn.svm import LinearSVC

model = LinearSVC()


#3. 컴파일, 훈련
model.fit(x_train, y_train)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 다중분류에서 loss 는 categorical_crossentropy

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)

# model.fit(x_train, y_train, epochs=100, batch_size=8, callbacks=[es], validation_split=0.3, verbose=2)

#4. 평가, 예측
result = model.score(x_test, y_test) # 어큐러씨가 나옴
print('model.score : ', result)

# loss = model.evaluate(x_test, y_test)

# print('loss : ', loss[0])
# print('accuracy : ', loss[1])

from sklearn.metrics import r2_score, accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuray_score : ', acc)

# 예측
print(y_test[:5])
y_predict2 = model.predict(x_test[:5])
print(y_predict)



