from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 데이터
datasets = load_boston()
print(datasets.feature_names)
'''
['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']

'ZN', 'CHAS', 'RAD' 삭제
'''

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

df.drop(['ZN', 'CHAS', 'RAD']  , inplace=True, axis=1)
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
random
r2 :  0.9206412337725959
[0.04083111 0.00095258 0.00753437 0.00120977 0.02415076 0.40406825
 0.0152267  0.06221644 0.00388852 0.01529159 0.01509583 0.01237516
 0.39715892]

삭제 후
[506 rows x 10 columns]
r2 :  0.9241620418440324 / random
[0.03856106 0.00657848 0.02454215 0.37503486 0.0139916  0.06560916
 0.01678326 0.01903192 0.01295893 0.42690858]
'''