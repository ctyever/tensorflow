from sklearn import datasets
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터 구성

datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=66)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(n_estimators=300, learning_rate=0.01, n_jobs=1)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric='logloss', #, 'logloss'
            eval_set=[(x_train, y_train), (x_test, y_test)]
)

# 4. 평가
result = model.score(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

# 평가 지표 logloss
# result :  0.9707602339181286
# acc :  0.9707602339181286

