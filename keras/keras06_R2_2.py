from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np


# 완성한뒤, 출력결과 스샷

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 4, 3, 5])

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=3)

loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x)
print('x의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)

print("r2스코어 : ", r2)

# 과제 2
# R2를 0.9 이상으로 올려라!

'''
epochs = 100, batch =3, loss :  0.38328155875205994, r2스코어 :  0.8083592193354378
1000, 3, loss :  0.3802139461040497, r2스코어 :  0.8098930159394214
1000, 2, loss :  0.3851162791252136, r2스코어 :  0.8074418662025721
1000, 2, layer수정, loss :  0.3836754262447357 , r2스코어 :  0.8081622762758457
10000, 2, loss :  0.3801063597202301 , r2스코어 :  0.8099468282588063
10000, 3, loss :  0.380046546459198 , r2스코어 :  0.8099767420029409
10000, 3, layer수정, loss :  0.3826726973056793, r2스코어 :  0.8086636544102745
test, train 나누고 더 안좋아짐

'''