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
from sklearn.datasets import load_boston

# 1.데이터
datasets = load_boston()
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

        scores = cross_val_score(model, x, y, cv=kfold, scoring='r2')

        # model.fit(x_train, y_train)

        # y_predict = model.predict(x_test)
        # acc = accuracy_score(y_test, y_predict)
        print(name, scores, '평균 :', round(np.mean(scores), 4))
    except:
        # continue
        print(name, '은 없는 놈!!!')

'''
ARDRegression [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866] 평균 : 0.6985
AdaBoostRegressor [0.91135475 0.81224027 0.8006045  0.81595951 0.86702813] 평균 : 0.8414
BaggingRegressor [0.90067036 0.81915455 0.7990357  0.89214343 0.88553924] 평균 : 0.8593
BayesianRidge [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051] 평균 : 0.7038
CCA [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276] 평균 : 0.6471
DecisionTreeRegressor [0.71390868 0.70067623 0.76817605 0.7502445  0.78362588] 평균 : 0.7433
DummyRegressor [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 : -0.0135
ElasticNet [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354] 평균 : 0.6708
ElasticNetCV [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608] 평균 : 0.6565
ExtraTreeRegressor [0.81978098 0.69219376 0.51535997 0.70987701 0.77421314] 평균 : 0.7023
ExtraTreesRegressor [0.93486801 0.85573316 0.76565364 0.87738906 0.93080286] 평균 : 0.8729
GammaRegressor [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635] 평균 : -0.0136
GaussianProcessRegressor [-6.07310526 -5.51957093 -6.33482574 -6.36383476 -5.35160828] 평균 : -5.9286
GradientBoostingRegressor [0.9453892  0.83810771 0.82761459 0.88590609 0.92995679] 평균 : 0.8854
HistGradientBoostingRegressor [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226] 평균 : 0.8581
HuberRegressor [0.74400323 0.64244715 0.52848946 0.37100122 0.63403398] 평균 : 0.584
IsotonicRegression [nan nan nan nan nan] 평균 : nan
KNeighborsRegressor [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 평균 : 0.5286
KernelRidge [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555] 평균 : 0.6854
Lars [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384] 평균 : 0.6977
LarsCV [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854] 평균 : 0.6928
Lasso [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473] 평균 : 0.6657
LassoCV [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127] 평균 : 0.6779
LassoLars [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 : -0.0135
LassoLarsCV [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787] 평균 : 0.6965
LassoLarsIC [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009] 평균 : 0.713
LinearRegression [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 : 0.7128
LinearSVR [ 0.10558327  0.7443362   0.49362952 -0.58327282  0.01428952] 평균 : 0.1549
MLPRegressor [0.60882435 0.52446384 0.36218398 0.35335781 0.43712219] 평균 : 0.4572
MultiOutputRegressor 은 없는 놈!!!
MultiTaskElasticNet [nan nan nan nan nan] 평균 : nan
MultiTaskElasticNetCV [nan nan nan nan nan] 평균 : nan
MultiTaskLasso [nan nan nan nan nan] 평균 : nan
MultiTaskLassoCV [nan nan nan nan nan] 평균 : nan
NuSVR [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ] 평균 : 0.2295
OrthogonalMatchingPursuit [0.58276176 0.565867   0.48689774 0.51545117 0.52049576] 평균 : 0.5343
OrthogonalMatchingPursuitCV [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377] 평균 : 0.6578
PLSCanonical [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868] 평균 : -2.2096
PLSRegression [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313] 평균 : 0.6847
PassiveAggressiveRegressor [-0.09479261  0.18509325  0.09432206 -0.71878518  0.22083758] 평균 : -0.0627
PoissonRegressor [0.85659255 0.8189989  0.66691488 0.67998192 0.75195656] 평균 : 0.7549
RANSACRegressor [ 0.25217813 -2.20076955  0.62803069  0.57946195  0.68096301] 평균 : -0.012
RadiusNeighborsRegressor [nan nan nan nan nan] 평균 : nan
RandomForestRegressor [0.92598762 0.85073929 0.82452152 0.88200847 0.89212904] 평균 : 0.8751
RegressorChain 은 없는 놈!!!
Ridge [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776] 평균 : 0.7109
RidgeCV [0.81125292 0.80010535 0.58888304 0.64008984 0.72362912] 평균 : 0.7128
SGDRegressor [-1.04023453e+27 -1.23428730e+26 -4.48456483e+25 -8.53927734e+26
 -1.92557097e+26] 평균 : -4.50998747585932e+26
SVR [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554] 평균 : 0.1963
StackingRegressor 은 없는 놈!!!
TheilSenRegressor [0.78961629 0.71252463 0.59289616 0.55051395 0.72154358] 평균 : 0.6734
TransformedTargetRegressor [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 : 0.7128
TweedieRegressor [0.7492543  0.75457294 0.56286929 0.57989884 0.63242475] 평균 : 0.6558
VotingRegressor 은 없는 놈!!!
'''

