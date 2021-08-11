from sklearn.utils import all_estimators
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

datasets = load_iris()
x = datasets.data
y = datasets.target

# 2. 모델 구성

allAlgorithms = all_estimators(type_filter='classifier')
# print(len(allAlgorithms)) # 모델의 개수 : 41
# allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms)

kfold = KFold(n_splits=5, shuffle=True, random_state=66)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        scores = cross_val_score(model, x, y, cv=kfold)

        # model.fit(x_train, y_train)

        # y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        print(name, scores, '평균 :', round(np.mean(scores), 4))
    except:
        # continue
        print(name, '은 없는 놈!!!')

'''
AdaBoostClassifier [0.63333333 0.93333333 1.         0.9        0.96666667]
BaggingClassifier [0.93333333 0.96666667 1.         0.9        0.96666667]
BernoulliNB [0.3        0.33333333 0.3        0.23333333 0.3       ]
CalibratedClassifierCV [0.9        0.83333333 1.         0.86666667 0.96666667]
CategoricalNB [0.9        0.93333333 0.93333333 0.9        1.        ]
ClassifierChain 은 없는 놈!!!
ComplementNB [0.66666667 0.66666667 0.7        0.6        0.7       ]
DecisionTreeClassifier [0.93333333 0.96666667 1.         0.9        0.93333333]
DummyClassifier [0.3        0.33333333 0.3        0.23333333 0.3       ]
ExtraTreeClassifier [0.9        0.93333333 0.93333333 0.93333333 0.96666667]
ExtraTreesClassifier [0.93333333 0.96666667 1.         0.86666667 0.96666667]
GaussianNB [0.96666667 0.9        1.         0.9        0.96666667]
GaussianProcessClassifier [0.96666667 0.96666667 1.         0.9        0.96666667]
GradientBoostingClassifier [0.93333333 0.96666667 1.         0.93333333 0.96666667]
HistGradientBoostingClassifier [0.86666667 0.96666667 1.         0.9        0.96666667]
KNeighborsClassifier [0.96666667 0.96666667 1.         0.9        0.96666667]
LabelPropagation [0.93333333 1.         1.         0.9        0.96666667]
LabelSpreading [0.93333333 1.         1.         0.9        0.96666667]
LinearDiscriminantAnalysis [1.  1.  1.  0.9 1. ]
LinearSVC [0.96666667 0.96666667 1.         0.9        1.        ]
LogisticRegression [1.         0.96666667 1.         0.9        0.96666667]
LogisticRegressionCV [1.         0.96666667 1.         0.9        1.        ]
MLPClassifier [0.96666667 0.93333333 1.         0.93333333 1.        ]
MultiOutputClassifier 은 없는 놈!!!
MultinomialNB [0.96666667 0.93333333 1.         0.93333333 1.        ]
NearestCentroid [0.93333333 0.9        0.96666667 0.9        0.96666667]
NuSVC [0.96666667 0.96666667 1.         0.93333333 1.        ]
OneVsOneClassifier 은 없는 놈!!!
OneVsRestClassifier 은 없는 놈!!!
OutputCodeClassifier 은 없는 놈!!!
PassiveAggressiveClassifier [0.96666667 0.8        0.83333333 0.8        1.        ]
Perceptron [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ]
QuadraticDiscriminantAnalysis [1.         0.96666667 1.         0.93333333 1.        ]
RadiusNeighborsClassifier [0.96666667 0.9        0.96666667 0.93333333 1.        ]
RandomForestClassifier [0.93333333 0.96666667 1.         0.86666667 0.96666667]
RidgeClassifier [0.86666667 0.8        0.93333333 0.7        0.9       ]
RidgeClassifierCV [0.86666667 0.8        0.93333333 0.7        0.9       ]
SGDClassifier [0.6        0.93333333 0.7        0.86666667 0.73333333]
SVC [0.96666667 0.96666667 1.         0.93333333 0.96666667]
StackingClassifier 은 없는 놈!!!
VotingClassifier 은 없는 놈!!!
'''






