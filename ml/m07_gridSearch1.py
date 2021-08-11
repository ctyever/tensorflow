import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import datasets
from sklearn.datasets import load_iris

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # 디시전트리의 확장형 / decisiontree
from sklearn.metrics import r2_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

datasets = load_iris()
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
    {'C':[1, 10, 100, 1000], 'kernel':['linear']},
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'], 'gamma':[0.001, 0.0001]}
]

model = GridSearchCV(SVC(), parameters, cv=kfold)

# model = SVC()

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('최적의 매개변수 : ', model.best_estimator_)
print('best_score_ : ', model.best_score_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# best_score_ :  0.9619047619047618

print('model.score', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))

# model.score 1.0
# accuracy_score :  1.0







