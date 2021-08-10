from sklearn.utils import all_estimators
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성

allAlgorithms = all_estimators(type_filter='classifier')
# print(len(allAlgorithms)) # 모델의 개수 : 41
# allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        # continue
        print(name, '은 없는 놈!!!')

'''     
AdaBoostClassifier 의 정답률 :  1.0
BaggingClassifier 의 정답률 :  1.0
BernoulliNB 의 정답률 :  0.35555555555555557
CalibratedClassifierCV 의 정답률 :  0.9777777777777777
CategoricalNB 은 없는 놈!!!
ClassifierChain 은 없는 놈!!!
ComplementNB 의 정답률 :  0.6888888888888889
DecisionTreeClassifier 의 정답률 :  0.9777777777777777
DummyClassifier 의 정답률 :  0.28888888888888886
ExtraTreeClassifier 의 정답률 :  1.0
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  0.9777777777777777
GradientBoostingClassifier 의 정답률 :  1.0
HistGradientBoostingClassifier 의 정답률 :  1.0
KNeighborsClassifier 의 정답률 :  1.0
LabelPropagation 의 정답률 :  0.9777777777777777
LabelSpreading 의 정답률 :  0.9777777777777777
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  1.0
LogisticRegression 의 정답률 :  0.9777777777777777
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  0.9555555555555556
MultiOutputClassifier 은 없는 놈!!!
MultinomialNB 의 정답률 :  0.6888888888888889
NearestCentroid 의 정답률 :  1.0
NuSVC 의 정답률 :  1.0
OneVsOneClassifier 은 없는 놈!!!
OneVsRestClassifier 은 없는 놈!!!
OutputCodeClassifier 은 없는 놈!!!
PassiveAggressiveClassifier 의 정답률 :  1.0
Perceptron 의 정답률 :  0.8444444444444444
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.5111111111111111
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  0.9555555555555556
RidgeClassifierCV 의 정답률 :  0.9555555555555556
SGDClassifier 의 정답률 :  1.0
SVC 의 정답률 :  1.0
StackingClassifier 은 없는 놈!!!
VotingClassifier 은 없는 놈!!!
''' 




