from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터 구성

datasets = load_diabetes()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBRegressor(n_estimators=100, learning_rate=0.05, n_jobs=1)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric='rmse', # 'mae', 'logloss'
            eval_set=[(x_train, y_train), (x_test, y_test)]
)

# 4. 평가
result = model.score(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

# XGBRegressor
# result :  0.46058816182147155
# r2 :  0.46058816182147155

# save_model
# loss :  2650.04345703125
# r2스코어 :  0.5204645824352365

#체크포인트
# loss :  2797.805419921875
# r2스코어 :  0.49372649792464096

# print('예측값 : ', y_predict)