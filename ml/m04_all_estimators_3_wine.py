from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd

datasets = pd.read_csv('./data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)

datasets = datasets.to_numpy()
# print(datasets)

x = datasets[:, :11]
y = datasets[:, 11:]
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

# print(y_test)
# print(y_train)


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
AdaBoostClassifier 의 정답률 :  0.43197278911564624
BaggingClassifier 의 정답률 :  0.6244897959183674
BernoulliNB 의 정답률 :  0.4421768707482993      
CalibratedClassifierCV 의 정답률 :  0.5265306122448979
CategoricalNB 의 정답률 :  0.44285714285714284        
ClassifierChain 은 없는 놈!!!
ComplementNB 의 정답률 :  0.38571428571428573
DecisionTreeClassifier 의 정답률 :  0.5877551020408164
DummyClassifier 의 정답률 :  0.4421768707482993
ExtraTreeClassifier 의 정답률 :  0.5829931972789115
ExtraTreesClassifier 의 정답률 :  0.6578231292517007
GaussianNB 의 정답률 :  0.45170068027210886
GaussianProcessClassifier 의 정답률 :  0.5265306122448979
GradientBoostingClassifier 의 정답률 :  0.5680272108843537
HistGradientBoostingClassifier 의 정답률 :  0.6408163265306123
KNeighborsClassifier 의 정답률 :  0.5435374149659864
LabelPropagation 의 정답률 :  0.5176870748299319
LabelSpreading 의 정답률 :  0.5142857142857142
LinearDiscriminantAnalysis 의 정답률 :  0.5170068027210885
LinearSVC 의 정답률 :  0.5319727891156463
LogisticRegression 의 정답률 :  0.5231292517006803
LogisticRegressionCV 의 정답률 :  0.5163265306122449
MLPClassifier 의 정답률 :  0.5244897959183673
MultiOutputClassifier 은 없는 놈!!!
MultinomialNB 의 정답률 :  0.4421768707482993
NearestCentroid 의 정답률 :  0.32040816326530613
NuSVC 은 없는 놈!!!
OneVsOneClassifier 은 없는 놈!!!
OneVsRestClassifier 은 없는 놈!!!
OutputCodeClassifier 은 없는 놈!!!
PassiveAggressiveClassifier 의 정답률 :  0.3891156462585034
Perceptron 의 정답률 :  0.43605442176870746
QuadraticDiscriminantAnalysis 의 정답률 :  0.482312925170068
RadiusNeighborsClassifier 의 정답률 :  0.4435374149659864
RandomForestClassifier 의 정답률 :  0.6557823129251701
RidgeClassifier 의 정답률 :  0.5272108843537415
RidgeClassifierCV 의 정답률 :  0.527891156462585
SGDClassifier 의 정답률 :  0.5258503401360545
SVC 의 정답률 :  0.5340136054421769
StackingClassifier 은 없는 놈!!!
VotingClassifier 은 없는 놈!!!
'''