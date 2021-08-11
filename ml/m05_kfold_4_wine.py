from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd

from sklearn.utils import all_estimators
from sklearn.metrics import r2_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

datasets = pd.read_csv('./data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)

datasets = datasets.to_numpy()
# print(datasets)

x = datasets[:, :11]
y = datasets[:, 11:]
# print(y)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)


# 2. 모델 구성

# model = LinearSVC()
# Acc:  [0.38265306 0.44285714 0.18877551 0.51481103 0.44841675] 0.3955
# model = SVC()
# Acc:  [0.4622449  0.4377551  0.44693878 0.46373851 0.4473953 ] 0.4516
# model = KNeighborsClassifier()
# Acc:  [0.48979592 0.48469388 0.4755102  0.46373851 0.45863126] 0.4745
# model = LogisticRegression()
# Acc:  [0.47142857 0.45204082 0.44795918 0.48723187 0.46578141] 0.4649
# model = DecisionTreeClassifier()
# Acc:  [0.6377551  0.6        0.59795918 0.58835546 0.60674157] 0.6062
model = RandomForestClassifier()
# Acc:  [0.70714286 0.67653061 0.69387755 0.69765066 0.68437181] 0.6919

#3. 컴파일, 훈련
#4. 평가, 예측

scores = cross_val_score(model, x, y, cv=kfold)

print("Acc: ", scores, round(np.mean(scores), 4))


