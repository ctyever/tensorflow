import numpy as np 
from sklearn.model_selection import train_test_split

x = np.load('./_save/_npy/k59_7_x.npy')
y = np.load('./_save/_npy/k59_7_y.npy')

# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66
)

# print(x_train.shape, x_test.shape) # (371, 150, 150, 3) (159, 150, 150, 3)
# print(y_train.shape, y_test.shape) # (371, 2) (159, 2)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), padding='valid', activation='relu', input_shape=(150,150,3)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))
model.add(MaxPooling2D()) 

model.add(Conv2D(64, (2, 2), activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(64, (2, 2), padding='same', activation='relu')) 
model.add(MaxPooling2D()) 

model.add(Flatten()) 
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
es = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                    write_graph=True, write_images=True)

# model.fit(x_train, y_train)
hist = model.fit(x_train, y_train, epochs=100, steps_per_epoch=32,
    validation_split=0.1, callbacks=[es, tb]
) # 160/5 = 32

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위에거로 시각화 할 것

print('acc :', acc[-1])
print('val_acc: ', val_acc[:-1])

# 4. 평가, 예측

acc = model.evaluate(x_test, y_test)[1]
print('acc : ', acc)
# result = model.predict(pred)
# pred = np.argmax(result, axis = 1)
# print('예측값 : ', pred)

'''
1차
acc :  0.7106918096542358

'''