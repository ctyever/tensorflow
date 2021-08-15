from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터 구성

datasets = load_boston()
x = datasets['data']
y = datasets['target']

# print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8, shuffle=True, random_state=66)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBRegressor(n_estimators=10000, learning_rate=0.01, # n_jobs=2
                     tree_method='gpu_hist', 
                     predictor='gpu_predictor', # cpu_predictor
                     gpu_id=0
)

# 3. 훈련
import time

start_time = time.time()

model.fit(x_train, y_train, verbose=1, eval_metric='rmse', # 'mae', 'logloss'
            eval_set=[(x_train, y_train), (x_test, y_test)]
)
print("걸린 시간 : ", time.time() - start_time)

'''
cpu
n_jobs=1 걸린 시간 : 9.222333192825317 / i7-9700
n_jobs=2 걸린 시간 :  7.73043155670166
n_jobs=4 걸린 시간 :  6.990024566650391
n_jobs=8 걸린 시간 :  6.944802284240723

gpu 
걸린 시간 :  37.95764112472534
'''




