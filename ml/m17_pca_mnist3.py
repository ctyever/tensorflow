import numpy as np 
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
# print(x.shape) # (70000, 28, 28)
x = x.reshape(70000, 28*28)


pca = PCA(n_components=784) # 
x = pca.fit_transform(x)
# print(x.shape) # 

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.999)+1) # 0.95 / 154, 0.99 / 331, 0.999 / 486

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()
# x_train = x[:60000]
# x_test = x[60000:]
# print(x_train.shape)
# print(x_test.shape)


# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()
# # scaler = StandardScaler()
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# # scaler = QuantileTransformer()
# # scaler = PowerTransformer()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(60000, 154, 1, 1)
# x_test = x_test.reshape(10000, 154, 1, 1)


# # 2. 모델 구성
# # tensorflow dnn 으로 구성하고... 기존 tensorflow dnn과 비교

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(y_train.shape, y_test.shape)

# # 2. 모델 구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D

# # model = Sequential()
# # model.add(Dense(100, input_shape=(154,)))
# # model.add(Dense(100, activation='relu'))
# # model.add(Dense(64, activation='relu')) 
# # model.add(Dense(64, activation='relu'))
# # # model.add(GlobalAveragePooling2D())
# # model.add(Dense(10, activation='softmax'))


# # model = Sequential()
# # model.add(Conv2D(filters=100, kernel_size=(2, 2), padding='same', input_shape=(154, 1, 1)))
# # # model.add(Conv2D(20, (2,2), activation='relu'))
# # # model.add(Conv2D(20, (2,2), activation='relu')) 
# # # model.add(MaxPooling2D()) 
# # model.add(Flatten()) 
# # model.add(Dense(64, activation='relu')) 
# # model.add(Dense(32, activation='relu'))
# # model.add(Dense(10, activation='softmax'))


# # model.summary()

# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류에서 loss 는 categorical_crossentropy

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

# import time
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=150, callbacks=[es], validation_split=0.1, verbose=2)
# end_time = time.time() - start_time

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss[0])
# print('accuracy : ', loss[1])

# pca 154 일 때 /cnn
# loss :  0.2892109453678131
# accuracy :  0.9610000252723694

#minmax
# loss :  0.1183680072426796
# accuracy :  0.9696999788284302

# pca 154 일 때
# loss :  0.20758706331253052
# accuracy :  0.9666000008583069

# pca 784 일 때
# loss :  0.26842957735061646
# accuracy :  0.9624999761581421


# loss :  0.12132221460342407
# accuracy :  0.984499990940094

# minmax 처리, batch_size=32 -> batch_size=150
# loss :  0.0800686776638031
# accuracy :  0.9868000149726868

# dnn 
# loss :  0.1577836275100708
# accuracy :  0.9747999906539917

# gap 처리
# loss :  0.29182305932044983
# accuracy :  0.916100025177002