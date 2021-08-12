from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_wine()
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.7, random_state=66
)

# 2. 모델
model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()

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
Number of Attributes: 13 numeric, predictive attributes and the class
class:
            - class_0
            - class_1
            - class_2
decision
acc :  0.9814814814814815
[0.0055201  0.         0.         0.         0.01830348 0.
 0.03003649 0.         0.02374506 0.12477838 0.01626976 0.37054514
 0.41080159]

random
acc :  1.0
[0.10933317 0.04018149 0.01497836 0.03953341 0.03503248 0.06469766
 0.14493206 0.00888966 0.03048972 0.11935825 0.10255025 0.11164388
 0.17837959]
'''

