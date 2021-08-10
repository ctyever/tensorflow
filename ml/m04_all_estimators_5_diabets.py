from sklearn.svm import LinearSVC, SVC # 먹히는지 확인
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

from icecream import ic
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn import datasets 
from sklearn.datasets import load_diabetes

# 1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# ic(x.shape, y.shape)  # x.shape: (442, 10), y.shape: (442,)
# ic(datasets.feature_names)  # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
# ic(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
# scaler = MinMaxScaler()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(y)

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


# MinMaxScaler
# loss :  1990.69287109375
# r2스코어 :  0.6397765814915889

'''
ARDRegression 의 정답률 :  0.6002021431737925
AdaBoostRegressor 의 정답률 :  0.49516543081034614
BaggingRegressor 의 정답률 :  0.3988254349888045
BayesianRidge 의 정답률 :  0.6035719100496502
CCA 의 정답률 :  0.5825461541443026
DecisionTreeRegressor 의 정답률 :  -0.012097664510511308
DummyRegressor 의 정답률 :  -0.010307682180093813
ElasticNet 의 정답률 :  -0.0005283687487036559
ElasticNetCV 의 정답률 :  0.530835820090738
ExtraTreeRegressor 의 정답률 :  -0.028244728821992915
ExtraTreesRegressor 의 정답률 :  0.5036905789743749
GammaRegressor 의 정답률 :  -0.0026356939743901187
GaussianProcessRegressor 의 정답률 :  -15.278736896635781
GradientBoostingRegressor 의 정답률 :  0.5016909208106629
HistGradientBoostingRegressor 의 정답률 :  0.4800369957137307
HuberRegressor 의 정답률 :  0.595320422727629
IsotonicRegression 은 없는 놈!!!
KNeighborsRegressor 의 정답률 :  0.4730968015300411
KernelRidge 의 정답률 :  -3.556723376973336
Lars 의 정답률 :  0.5900352656383726
LarsCV 의 정답률 :  0.6007838659835665
Lasso 의 정답률 :  0.40131447906114415
LassoCV 의 정답률 :  0.6016292025063978
LassoLars 의 정답률 :  0.45468470967039465
LassoLarsCV 의 정답률 :  0.6007838659835665
LassoLarsIC 의 정답률 :  0.6027649441842138
LinearRegression 의 정답률 :  0.5900352656383733
LinearSVR 의 정답률 :  -0.35446189710236364
MLPRegressor 의 정답률 :  -2.968485038355
MultiOutputRegressor 은 없는 놈!!!
MultiTaskElasticNet 은 없는 놈!!!
MultiTaskElasticNetCV 은 없는 놈!!!
MultiTaskLasso 은 없는 놈!!!
MultiTaskLassoCV 은 없는 놈!!!
NuSVR 의 정답률 :  0.16247975373511803
OrthogonalMatchingPursuit 의 정답률 :  0.3443972776662051
OrthogonalMatchingPursuitCV 의 정답률 :  0.5950203281004389
PLSCanonical 의 정답률 :  -1.256553805294621
PLSRegression 의 정답률 :  0.6104705521001149
PassiveAggressiveRegressor 의 정답률 :  0.5567160301933565
PoissonRegressor 의 정답률 :  0.404146586839525
RANSACRegressor 의 정답률 :  0.33561049900419215
RadiusNeighborsRegressor 의 정답률 :  -0.010307682180093813
RandomForestRegressor 의 정답률 :  0.5022016537932268
RegressorChain 은 없는 놈!!!
Ridge 의 정답률 :  0.488206754603569
RidgeCV 의 정답률 :  0.5995202529204973
SGDRegressor 의 정답률 :  0.47326024465079986
SVR 의 정답률 :  0.1837659735784033
StackingRegressor 은 없는 놈!!!
TheilSenRegressor 의 정답률 :  0.599693003381764
TransformedTargetRegressor 의 정답률 :  0.5900352656383733
TweedieRegressor 의 정답률 :  -0.0029402270467873137
VotingRegressor 은 없는 놈!!!
'''

