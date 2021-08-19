from numpy.lib.function_base import average
from sklearn import datasets
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings('ignore')

datasets = pd.read_csv('./data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)
datasets = datasets.values

x = datasets[:, :11]
y = datasets[:, 11]

# print(x.shape, y.shape) # (4898, 11) (4898,)

# print(pd.Series(y).value_counts())
'''
6.0    2198
5.0    1457
7.0     880
8.0     175
4.0     163
3.0      20
9.0       5
'''

################## 라벨 대통합 ###############################
print("=======================================")
'''
for index, value in enumerate(y):
    if value == 9:
        y[index] = 8
'''

for i in range(len(y)):
    if y[i] == 9:
        y[i] = 7
    elif y[i] == 8:
        y[i] = 7
    elif y[i] == 3:
        y[i] = 5
    elif y[i] == 4:
        y[i] = 5
    

# print(pd.Series(y).value_counts())

'''
6.0    2198
5.0    1457
7.0     880
8.0     180
4.0     163
3.0      20
'''

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=66, stratify=y
)
# print(pd.Series(y_train).value_counts())
# 6.0    1648
# 5.0    1093
# 7.0     660
# 8.0     131
# 4.0     122
# 3.0      15
# 9.0       4

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("model.score : ", score)  # model.score :  0.7004081632653061

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score : ", f1)

############## smote 적용 ####################
# print("============ smote 적용 =======================")

smote = SMOTE(random_state=66, k_neighbors=60)

start_time = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train, )
end_time = time.time() - start_time

print(pd.Series(y_smote_train).value_counts())
'''
============ smote 적용 =======================
0    53
1    53
2    53
'''
# print(x_smote_train.shape, y_smote_train.shape) # (159, 13) (159,)

print("smote 전 : ", x_train.shape, y_train.shape)
print("smote 후 : ", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 : \n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 : \n", pd.Series(y_smote_train).value_counts())
'''
smote 전 :  (111, 13) (111,)
smote 후 :  (159, 13) (159,)
smote전 레이블 값 분포 :
 1    53
0    44
2    14
dtype: int64
smote후 레이블 값 분포 :
 0    53
1    53
2    53
'''
model2 = XGBClassifier(n_jobs=-1)

model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
print("model2.score : ", score2)  # model2.score :  0.7093877551020408

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred,  average='macro')
print("f1_score : ", f1)








