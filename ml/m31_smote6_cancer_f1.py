# 지표는 f1

from sklearn import datasets
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) # (569, 30) (569,)

# print(pd.Series(y).value_counts())
# 1    357
# 0    212


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=66, stratify=y
)
# print(pd.Series(y_train).value_counts())

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("model.score : ", score)  # model2.score :  0.951048951048951

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred)
print("f1_score : ", f1) # f1_score :  0.9608938547486034  

############## smote 적용 ####################
print("============ smote 적용 =======================")

smote = SMOTE(random_state=66, k_neighbors=60)

start_time = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train, y_train, )
end_time = time.time() - start_time

print(pd.Series(y_smote_train).value_counts())
'''
============ smote 적용 =======================
0    267
1    267
'''
# print(x_smote_train.shape, y_smote_train.shape) # (159, 13) (159,)

print("smote 전 : ", x_train.shape, y_train.shape)
print("smote 후 : ", x_smote_train.shape, y_smote_train.shape)
print("smote전 레이블 값 분포 : \n", pd.Series(y_train).value_counts())
print("smote후 레이블 값 분포 : \n", pd.Series(y_smote_train).value_counts())
'''
smote 전 :  (426, 30) (426,)
smote 후 :  (534, 30) (534,)
smote전 레이블 값 분포 :
 1    267
0    159
dtype: int64
smote후 레이블 값 분포 :
 0    267
1    267
'''
model2 = XGBClassifier(n_jobs=-1)

model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score2 = model2.score(x_test, y_test)
print("model2.score : ", score2)  # model2.score :  0.951048951048951

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred)
print("f1_score : ", f1) # f1_score :  0.9608938547486034
