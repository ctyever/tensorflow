from sklearn import datasets
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터 구성

datasets = load_wine()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, shuffle=True, random_state=9)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(n_estimators=100, learning_rate=0.01, n_jobs=1)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric='mlogloss', #, 'logloss'
            eval_set=[(x_train, y_train), (x_test, y_test)]
)

# 4. 평가
result = model.score(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

# 평가 지표 mlogloss
# result :  0.9722222222222222
# acc :  0.9722222222222222
