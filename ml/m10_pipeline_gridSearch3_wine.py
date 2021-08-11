# 실습

# 모델 : RandomForestClassifier
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn import datasets
from sklearn.datasets import load_breast_cancer

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # 디시전트리의 확장형 / decisiontree
from sklearn.metrics import r2_score, accuracy_score
import warnings
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

datasets = pd.read_csv('./data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)

datasets = datasets.to_numpy()
# print(datasets)

x = datasets[:, :11]
y = datasets[:, 11:]

# print(x.shape, y.shape) # (150, 4) (150,)
# print(y) 
x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# 2. 모델 구성
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
pipe = Pipeline([("scaler", MinMaxScaler()), ("rf", RandomForestClassifier())])


parameters = [
    {'rf__n_estimators' : [100, 200], 'rf__max_depth' : [6, 9, 11, 13], 'rf__min_samples_leaf' : [3, 5, 7, 10]},
    {'rf__max_depth' : [6, 8, 10, 12], 'rf__min_samples_split' : [2, 3, 5, 10]},
    {'rf__min_samples_leaf' : [3, 5, 7, 10]},
    {'rf__min_samples_split' : [2, 3, 5, 10]},
    # {'n_jobs' : [-1, 2, 4]}
]

model = GridSearchCV(pipe, parameters, cv=kfold, verbose=1)

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('최적의 매개변수 : ', model.best_estimator_)
print('best_params_ : ', model.best_params_)
print('best_score_ : ', model.best_score_)

print('model.score', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))

# 최적의 매개변수 :  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
#                 ('randomforestclassifier',
#                  RandomForestClassifier(min_samples_split=3))])
# best_score_ :  0.6656968355642571
# model.score 0.6496598639455783
# accuracy_score :  0.6496598639455783

# 최적의 매개변수 :  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('rf', RandomForestClassifier(min_samples_split=5))])
# best_params_ :  {'rf__min_samples_split': 5}
# best_score_ :  0.6674478091549446
# model.score 0.6503401360544218
# accuracy_score :  0.6503401360544218







