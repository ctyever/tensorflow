from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn import datasets  
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(y[:100])

# print(x.shape, y.shape) # (569, 30) (569,)

# print(np.unique(y)) # (0, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

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
AdaBoostClassifier 의 정답률 :  0.9649122807017544
BaggingClassifier 의 정답률 :  0.9590643274853801
BernoulliNB 의 정답률 :  0.631578947368421       
CalibratedClassifierCV 의 정답률 :  0.9707602339181286
CategoricalNB 은 없는 놈!!!
ClassifierChain 은 없는 놈!!!
ComplementNB 의 정답률 :  0.8596491228070176
DecisionTreeClassifier 의 정답률 :  0.9415204678362573
DummyClassifier 의 정답률 :  0.6374269005847953
ExtraTreeClassifier 의 정답률 :  0.9005847953216374
ExtraTreesClassifier 의 정답률 :  0.9649122807017544
GaussianNB 의 정답률 :  0.9298245614035088
GaussianProcessClassifier 의 정답률 :  0.9707602339181286
GradientBoostingClassifier 의 정답률 :  0.9532163742690059
HistGradientBoostingClassifier 의 정답률 :  0.9707602339181286
KNeighborsClassifier 의 정답률 :  0.9649122807017544
LabelPropagation 의 정답률 :  0.9766081871345029
LabelSpreading 의 정답률 :  0.9707602339181286
LinearDiscriminantAnalysis 의 정답률 :  0.9649122807017544
LinearSVC 의 정답률 :  0.9766081871345029
LogisticRegression 의 정답률 :  0.9766081871345029
LogisticRegressionCV 의 정답률 :  0.9766081871345029
MLPClassifier 의 정답률 :  0.9824561403508771
MultiOutputClassifier 은 없는 놈!!!
MultinomialNB 의 정답률 :  0.8421052631578947
NearestCentroid 의 정답률 :  0.9473684210526315
NuSVC 의 정답률 :  0.9590643274853801
OneVsOneClassifier 은 없는 놈!!!
OneVsRestClassifier 은 없는 놈!!!
OutputCodeClassifier 은 없는 놈!!!
PassiveAggressiveClassifier 의 정답률 :  0.9590643274853801
Perceptron 의 정답률 :  0.9707602339181286
QuadraticDiscriminantAnalysis 의 정답률 :  0.9590643274853801
RadiusNeighborsClassifier 은 없는 놈!!!
RandomForestClassifier 의 정답률 :  0.9590643274853801
RidgeClassifier 의 정답률 :  0.9707602339181286
RidgeClassifierCV 의 정답률 :  0.9649122807017544
SGDClassifier 의 정답률 :  0.9766081871345029
SVC 의 정답률 :  0.9824561403508771
StackingClassifier 은 없는 놈!!!
VotingClassifier 은 없는 놈!!!
'''

