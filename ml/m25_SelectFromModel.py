from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel

# 1. 데이터
# datasets = load_boston()
# x = datasets.data
# y = datasets.target
x, y = load_boston(return_X_y=True) # 이런 방식도 있음, 중요하진 않음

# print(x.shape, y.shape) # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=66
)

# 2. 모델
model = XGBRegressor(n_jobs=8)

# 3. 출력
model.fit(x_train, y_train)

# 4. 평가, 예측
score = model.score(x_test, y_test)
# print('model.score : ', score)  # model.score :  0.9221188601856797

thresholds = np.sort(model.feature_importances_)
# print(thresholds)
'''
[0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
 0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
 0.42848358]
'''

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # print(selection)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))

'''
(404, 13) (102, 13)
Thresh=0.001, n=13, R2: 92.21%
(404, 12) (102, 12)
Thresh=0.004, n=12, R2: 92.16%
(404, 11) (102, 11)
Thresh=0.012, n=11, R2: 92.03%
(404, 10) (102, 10)
Thresh=0.012, n=10, R2: 92.19%
(404, 9) (102, 9)
Thresh=0.014, n=9, R2: 93.08%
(404, 8) (102, 8)
Thresh=0.015, n=8, R2: 92.37%
(404, 7) (102, 7)
Thresh=0.018, n=7, R2: 91.48%
(404, 6) (102, 6)
Thresh=0.030, n=6, R2: 92.71%
(404, 5) (102, 5)
Thresh=0.042, n=5, R2: 91.74%
(404, 4) (102, 4)
Thresh=0.052, n=4, R2: 92.11%
(404, 3) (102, 3)
Thresh=0.069, n=3, R2: 92.52%
(404, 2) (102, 2)
Thresh=0.301, n=2, R2: 69.41%
(404, 1) (102, 1)
Thresh=0.428, n=1, R2: 44.98%
'''

