from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor


# 1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.7, random_state=66
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

import matplotlib.pyplot as plt
import numpy as np 

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

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
'''