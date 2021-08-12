from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor
import pandas as pd


# 1. 데이터
datasets = load_breast_cancer()
# print(datasets.DESCR)
# print(datasets.feature_names)
'''
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']

worst compactness, concavity error, mean fractal dimension, mean symmetry, mean perimeter, worst symmetry
'''

df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])

df.drop(['worst compactness', 'concavity error', 'mean fractal dimension', 'mean symmetry', 'mean perimeter', 'worst symmetry']  , inplace=True, axis=1)
# print(df)
x = df.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
    x, datasets.target, train_size=0.7, random_state=66
)

# 2. 모델
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)

print(model.feature_importances_)

# import matplotlib.pyplot as plt
# import numpy as np 

# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)
# plt.show()

'''
Number of Attributes: 30 numeric, predictive attributes and the class
decision
acc :  0.9415204678362573
[0.         0.02869086 0.         0.         0.00845356 0.01502854 
 0.         0.         0.         0.03043354 0.01481385 0.
 0.         0.         0.         0.         0.         0.                                                                                858891
 0.         0.         0.         0.01859782 0.         0.77570942 
 0.         0.         0.         0.10827241 0.         0.        ]

acc :  0.9590643274853801
[0.02866684 0.01644061 0.02338074 0.08873688 0.0041212  0.00618807
 0.01974947 0.11384283 0.00314591 0.00420995 0.02491115 0.00688549
 0.01295074 0.03174559 0.00348188 0.00523247 0.00405006 0.00367794
 0.00345435 0.0058141  0.16617195 0.02056112 0.12409546 0.15326213
 0.01712761 0.0106973  0.0197627  0.06617911 0.00415657 0.0072998 ]

acc :  0.9824561403508771 / XGBClassifier()
[0.03474978 0.03467545 0.         0.00751567 0.00292065 0.00687929
 0.01235841 0.02046652 0.         0.00054058 0.01655321 0.00162256
 0.01760868 0.00867776 0.01414008 0.00689266 0.00062871 0.00279585
 0.00357394 0.00587361 0.15269534 0.01300185 0.33814007 0.21980153
 0.00525721 0.         0.00569597 0.06131698 0.00166174 0.00395583]

6개 칼럼 삭제 / XGBClassifier()
acc :  0.9824561403508771
[0.03012681 0.02679937 0.00132736 0.00267793 0.00360012 0.03147845
 0.01084449 0.01101684 0.00082111 0.0153039  0.00701624 0.01224054
 0.00570793 0.00226558 0.00261272 0.00165795 0.21376093 0.01182101
 0.39163715 0.1529427  0.00631116 0.0047506  0.04875922 0.00451987]

'''