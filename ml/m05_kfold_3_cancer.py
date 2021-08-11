import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import datasets
from sklearn.datasets import load_breast_cancer

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  

import warnings

warnings.filterwarnings('ignore')

datasets = load_breast_cancer()
# print(datasets.DESCR)
# print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets.target

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# 2. 모델 구성

model = LinearSVC()
# Acc:  [0.90350877 0.93859649 0.89473684 0.92982456 0.90265487] 0.9139
# model = SVC()
# Acc:  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 0.921
# model = KNeighborsClassifier()
# Acc:  [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 0.928
# model = LogisticRegression()
# Acc:  [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177] 0.9385
# model = DecisionTreeClassifier()
# Acc:  [0.94736842 0.9122807  0.93859649 0.88596491 0.94690265] 0.9262
# model = RandomForestClassifier()
# Acc:  [0.96491228 0.96491228 0.97368421 0.94736842 0.97345133] 0.9649


#3. 컴파일, 훈련
#4. 평가, 예측

scores = cross_val_score(model, x, y, cv=kfold)

print("Acc: ", scores, round(np.mean(scores), 4))






