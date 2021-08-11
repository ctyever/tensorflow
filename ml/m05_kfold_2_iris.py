import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import datasets
from sklearn.datasets import load_iris

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # 디시전트리의 확장형 / decisiontree

import warnings

warnings.filterwarnings('ignore')

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

# print(x.shape, y.shape) # (150, 4) (150,)
# print(y) 
'''
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
 '''
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# 2. 모델 구성

model = LinearSVC()
# Acc:  Acc:  [0.85714286 1.         0.95238095 0.95238095 0.95238095] 0.9429
# model = SVC()
# Acc:  [0.9047619  0.95238095 1.         1.         0.9047619 ] 0.9524
# model = KNeighborsClassifier()
# Acc:  [0.85714286 1.         0.95238095 0.95238095 0.95238095] 0.9429
# model = LogisticRegression()
# Acc:  [0.85714286 1.         1.         1.         0.95238095] 0.9619
# model = DecisionTreeClassifier()
# Acc:  [0.85714286 0.95238095 0.9047619  0.95238095 0.80952381] 0.8952
# model = RandomForestClassifier()
# Acc:  [0.85714286 0.95238095 0.9047619  0.95238095 0.9047619 ] 0.9143


#3. 컴파일, 훈련
#4. 평가, 예측

scores = cross_val_score(model, x_train, y_train, cv=kfold)

print("Acc: ", scores, round(np.mean(scores), 4))






