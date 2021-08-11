# 실습

# 모델 : RandomForestClassifier

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn import datasets
from sklearn.datasets import load_diabetes

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # 디시전트리의 확장형 / decisiontree
from sklearn.metrics import r2_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

datasets = load_diabetes()
# print(datasets.DESCR)
# print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (150, 4) (150,)
# print(y) 
x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# 2. 모델 구성

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 9, 11, 13], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    # {'n_jobs' : [-1, 2, 4]}
]

# model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1)
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1)
# model = SVC(C=1, kernel='linear')

# model = SVC()

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('최적의 매개변수 : ', model.best_estimator_)
print('best_score_ : ', model.best_score_)

#random 
# 최적의 매개변수 :  RandomForestRegressor(max_depth=13, min_samples_leaf=10, n_estimators=200)
# best_score_ :  0.41306017176745363

# grid
# 최적의 매개변수 :  RandomForestRegressor(max_depth=6, min_samples_leaf=10, n_estimators=200)
# best_score_ :  0.4149529334696556

print('model.score', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('r2_score : ', r2_score(y_test, y_predict))

#random
# model.score 0.5211997369670177
# r2_score :  0.5211997369670177

# grid
# model.score 0.5267336916504863
# r2_score :  0.5267336916504863







