from sklearn.svm import LinearSVC, SVC # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

#1. loss 와 r2로 평가
# MINMax 와 Standard 결과를 명시

from icecream import ic
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn import datasets
from sklearn.datasets import load_boston

# 1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

# ic(x.shape, y.shape)  # x.shape: (506, 13), y.shape: (506,)
# ic(datasets.feature_names) # datasets.feature_names: array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')
# ic(datasets.DESCR)

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

# allAlgorithms = all_estimators(type_filter='classifier')
# print(len(allAlgorithms)) # 모델의 개수 : 41
allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms)

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        acc = r2_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        # continue
        print(name, '은 없는 놈!!!')

'''
ARDRegression 의 정답률 :  0.7774752052825809
AdaBoostRegressor 의 정답률 :  0.8493663127093575
BaggingRegressor 의 정답률 :  0.8158383997520838
BayesianRidge 의 정답률 :  0.7807712494900036
CCA 의 정답률 :  0.7608263146559945
DecisionTreeRegressor 의 정답률 :  0.6773580878065861
DummyRegressor 의 정답률 :  -0.019031831720200065
ElasticNet 의 정답률 :  0.13936247669889124
ElasticNetCV 의 정답률 :  0.7785010149851183
ExtraTreeRegressor 의 정답률 :  0.705945757896845
ExtraTreesRegressor 의 정답률 :  0.8743514981385617
GammaRegressor 의 정답률 :  0.17070103098691047
GaussianProcessRegressor 의 정답률 :  -0.47529287183107183
GradientBoostingRegressor 의 정답률 :  0.888957507388401
HistGradientBoostingRegressor 의 정답률 :  0.8591369609489108
HuberRegressor 의 정답률 :  0.7447767594665695
IsotonicRegression 은 없는 놈!!!
KNeighborsRegressor 의 정답률 :  0.7905322932653436
KernelRidge 의 정답률 :  0.6857051874805018
Lars 의 정답률 :  0.7826126074271011
LarsCV 의 정답률 :  0.7826126074271011
Lasso 의 정답률 :  0.20836256003654374
LassoCV 의 정답률 :  0.7822518098368609
LassoLars 의 정답률 :  -0.019031831720200065
LassoLarsCV 의 정답률 :  0.7826126074271011
LassoLarsIC 의 정답률 :  0.77885143199499
LinearRegression 의 정답률 :  0.7826126074271011
LinearSVR 의 정답률 :  0.6209065288858684
MLPRegressor 의 정답률 :  0.19379409370366685
MultiOutputRegressor 은 없는 놈!!!
MultiTaskElasticNet 은 없는 놈!!!
MultiTaskElasticNetCV 은 없는 놈!!!
MultiTaskLasso 은 없는 놈!!!
MultiTaskLassoCV 은 없는 놈!!!
NuSVR 의 정답률 :  0.565226285689969
OrthogonalMatchingPursuit 의 정답률 :  0.5545805166503026
OrthogonalMatchingPursuitCV 의 정답률 :  0.7310024148963548
PLSCanonical 의 정답률 :  -1.5937922552021675
PLSRegression 의 정답률 :  0.746876771020851
PassiveAggressiveRegressor 의 정답률 :  0.7380409119671398
PoissonRegressor 의 정답률 :  0.6684892854705593
RANSACRegressor 의 정답률 :  0.5666350193560064
RadiusNeighborsRegressor 의 정답률 :  0.38585516618453164
RandomForestRegressor 의 정답률 :  0.8332585313303433
RegressorChain 은 없는 놈!!!
Ridge 의 정답률 :  0.7740811308897126
RidgeCV 의 정답률 :  0.7740811308897043
SGDRegressor 의 정답률 :  0.7545000930069354
SVR 의 정답률 :  0.547585817519747
StackingRegressor 은 없는 놈!!!
TheilSenRegressor 의 정답률 :  0.7333308274322647
TransformedTargetRegressor 의 정답률 :  0.7826126074271011
TweedieRegressor 의 정답률 :  0.17521272325976556
VotingRegressor 은 없는 놈!!!
'''

