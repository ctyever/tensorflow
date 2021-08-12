#실습
# 피처임포터스가 전체 중요도에서 20% 미만인 컬럼등을 제거하여 데이터셋 구성
# 각 모델별로 돌려서 결과 도출

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 데이터
datasets = load_iris()
# print(type(datasets.data)) # <class 'numpy.ndarray'>
# print(datasets.DESCR)
# print(datasets.data)
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
# print(df)
'''
   sepal length (cm) sepal width (cm) petal length (cm) petal width (cm)
0                 5.1              3.5               1.4              0.2
1                 4.9              3.0               1.4              0.2
2                 4.7              3.2               1.3              0.2
3                 4.6              3.1               1.5              0.2
4                 5.0              3.6               1.4              0.2
..                ...              ...               ...              ...
145               6.7              3.0               5.2              2.3
146               6.3              2.5               5.0              1.9
147               6.5              3.0               5.2              2.0
148               6.2              3.4               5.4              2.3
149               5.9              3.0               5.1              1.8
'''
df.drop('sepal width (cm)', inplace=True, axis=1)
# print(df)
x = df.to_numpy()
# print(x)
x_train, x_test, y_train, y_test = train_test_split(
    x, datasets.target, train_size=0.7, random_state=66
)

# 2. 모델
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)

print(model.feature_importances_)

# import matplotlib.pyplot as plt
# import numpy as np 

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)
# plt.show()

'''
- sepal length in cm
- sepal width in cm
- petal length in cm
- petal width in cm

acc :  0.9111111111111111
[0.02834522 0.783171   0.18848379]

sepal width 삭제
acc :  0.8888888888888888
[0.19734565 0.44814264 0.35451172]

원본
acc :  0.9111111111111111
[0.09265661 0.03503764 0.41548035 0.4568254 ]
'''