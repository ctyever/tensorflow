from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 데이터
datasets = load_diabetes()

# print(datasets.data.shape) # (442, 10)
print(datasets.feature_names)

'''
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

's1', 's2', 's4' 삭제
'''

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

df.drop(['s1', 's2', 's4']  , inplace=True, axis=1)
print(df)
x = df.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(
    x, datasets.target, train_size=0.8, random_state=66
)

# 2. 모델
# model = DecisionTreeRegressor(max_depth=4)
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
r2 = model.score(x_test, y_test)
print('r2 : ', r2)

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
decisiion
r2 :  0.33339660919782466
[0.03400704 0.         0.26623557 0.11279298 0.         0.
 0.01272153 0.         0.51986371 0.05437917]

[442 rows x 7 columns]
r2 :  0.33339660919782466 / decisiion
[0.03400704 0.         0.26748278 0.11279298 0.01272153 0.51986371
 0.05313196]      

random
r2 :  0.3724385626991801
[0.06418666 0.01278901 0.28290363 0.11247318 0.04177911 0.05436326
 0.04622855 0.02298191 0.28859948 0.0736952 ]

[442 rows x 7 columns]
r2 :  0.3905822362233916 / random
[0.0844715  0.01349963 0.2812048  0.13161472 0.06644838 0.33434287
 0.0884181 ]
'''