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
from sklearn.model_selection import KFold, cross_val_score


from sklearn import datasets
from sklearn.datasets import load_diabetes

# 1.데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# ic(x.shape, y.shape)  # x.shape: (506, 13), y.shape: (506,)
# ic(datasets.feature_names) # datasets.feature_names: array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')
# ic(datasets.DESCR)

# 2. 모델 구성
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# allAlgorithms = all_estimators(type_filter='classifier')
# print(len(allAlgorithms)) # 모델의 개수 : 41
allAlgorithms = all_estimators(type_filter='regressor')
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
ARDRegression [0.49874835 0.48765748 0.56284846 0.37728801 0.53474369] 평균 : 0.4923
AdaBoostRegressor [0.37506344 0.45461127 0.48522568 0.40208568 0.4374624 ] 평균 : 0.4309
BaggingRegressor [0.30306459 0.37193007 0.37566314 0.29377866 0.35541012] 평균 : 0.34
BayesianRidge [0.50082189 0.48431051 0.55459312 0.37600508 0.5307344 ] 평균 : 0.4893
CCA [0.48696409 0.42605855 0.55244322 0.21708682 0.50764701] 평균 : 0.438
DecisionTreeRegressor [-0.28416991 -0.17811718 -0.06232472 -0.06126545  0.08974352] 평균 : -0.0992
DummyRegressor [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] 평균 : -0.0033
ElasticNet [ 0.00810127  0.00637294  0.00924848  0.0040621  -0.00081988] 평균 : 0.0054
ElasticNetCV [0.43071558 0.461506   0.49133954 0.35674829 0.4567084 ] 평균 : 0.4394
ExtraTreeRegressor [-0.27332529 -0.05374065 -0.04358294 -0.02887366 -0.30999898] 평균 : -0.1419
ExtraTreesRegressor [0.3835694  0.46995073 0.52497705 0.4067239  0.4654994 ] 평균 : 0.4501
GammaRegressor [ 0.00523561  0.00367973  0.0060814   0.00174734 -0.00306898] 평균 : 0.0027
GaussianProcessRegressor [ -5.6360757  -15.27401119  -9.94981439 -12.46884878 -12.04794389] 평균 : -11.0753
GradientBoostingRegressor [0.39181002 0.47912808 0.47993572 0.39203506 0.44408693] 평균 : 0.4374
HistGradientBoostingRegressor [0.28899498 0.43812684 0.51713242 0.37267554 0.35643755] 평균 : 0.3947
HuberRegressor [0.50334705 0.47508237 0.54650576 0.36883712 0.5173073 ] 평균 : 0.4822
IsotonicRegression [nan nan nan nan nan] 평균 : nan
KNeighborsRegressor [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969] 평균 : 0.3673
KernelRidge [-3.38476443 -3.49366182 -4.0996205  -3.39039111 -3.60041537] 평균 : -3.5938
Lars [ 0.49198665 -0.66475442 -1.04410299 -0.04236657  0.51190679] 평균 : -0.1495
LarsCV [0.4931481  0.48774421 0.55427158 0.38001456 0.52413596] 평균 : 0.4879
Lasso [0.34315574 0.35348212 0.38594431 0.31614536 0.3604865 ] 평균 : 0.3518
LassoCV [0.49799859 0.48389346 0.55926851 0.37740074 0.51636393] 평균 : 0.487
LassoLars [0.36543887 0.37812653 0.40638095 0.33639271 0.38444891] 평균 : 0.3742
LassoLarsCV [0.49719648 0.48426377 0.55975856 0.37984022 0.51190679] 평균 : 0.4866
LassoLarsIC [0.49940515 0.49108789 0.56130589 0.37942384 0.5247894 ] 평균 : 0.4912
LinearRegression [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 평균 : 0.4876
LinearSVR [-0.33470258 -0.31629592 -0.41886516 -0.30276155 -0.47808399] 평균 : -0.3701
MLPRegressor [-2.71772778 -3.02636639 -3.24504145 -2.87330969 -3.17443549] 평균 : -3.0074
MultiOutputRegressor 은 없는 놈!!!
MultiTaskElasticNet [nan nan nan nan nan] 평균 : nan
MultiTaskElasticNetCV [nan nan nan nan nan] 평균 : nan
MultiTaskLasso [nan nan nan nan nan] 평균 : nan
MultiTaskLassoCV [nan nan nan nan nan] 평균 : nan
NuSVR [0.14471275 0.17351835 0.18539957 0.13894135 0.1663745 ] 평균 : 0.1618
OrthogonalMatchingPursuit [0.32934491 0.285747   0.38943221 0.19671679 0.35916077] 평균 : 0.3121
OrthogonalMatchingPursuitCV [0.47845357 0.48661326 0.55695148 0.37039612 0.53615516] 평균 : 0.4857
PLSCanonical [-0.97507923 -1.68534502 -0.8821301  -1.33987816 -1.16041996] 평균 : -1.2086
PLSRegression [0.47661395 0.4762657  0.5388494  0.38191443 0.54717873] 평균 : 0.4842
PassiveAggressiveRegressor [0.42827383 0.4895567  0.46716991 0.34377687 0.45163233] 평균 : 0.4361
PoissonRegressor [0.32061441 0.35803358 0.3666005  0.28203414 0.34340626] 평균 : 0.3341
RANSACRegressor [-0.03820775  0.25986449  0.39315051 -0.63030618  0.10365912] 평균 : 0.0176
RadiusNeighborsRegressor [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] 평균 : -0.0033
RandomForestRegressor [0.37149577 0.49410406 0.45149226 0.37841579 0.40327776] 평균 : 0.4198
RegressorChain 은 없는 놈!!!
Ridge [0.40936669 0.44788406 0.47057299 0.34467674 0.43339091] 평균 : 0.4212
RidgeCV [0.49525464 0.48761091 0.55171354 0.3801769  0.52749194] 평균 : 0.4884
SGDRegressor [0.39334215 0.44184732 0.46468424 0.32960417 0.41498719] 평균 : 0.4089
SVR [0.14331635 0.18438697 0.17864042 0.1424597  0.1468719 ] 평균 : 0.1591
StackingRegressor 은 없는 놈!!!
TheilSenRegressor [0.51540598 0.46665316 0.54983747 0.34209509 0.50770332] 평균 : 0.4763
TransformedTargetRegressor [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 평균 : 0.4876
TweedieRegressor [ 0.00585525  0.00425899  0.00702558  0.00183408 -0.00315042] 평균 : 0.0032
VotingRegressor 은 없는 놈!!!
'''



