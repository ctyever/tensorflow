import numpy as np
from sklearn import datasets 
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (442, 10) (442,)

pca = PCA(n_components=6) # 컬럼을 7개로 압축
x = pca.fit_transform(x)
# print(x.shape) # (442, 7)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
print(cumsum)
'''
[0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
 0.94794364 0.99131196 0.99914395 1.        ]
'''

print(np.argmax(cumsum >= 0.94)+1)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)

# 2. 모델
from xgboost import XGBRegressor
model = XGBRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 : ', result)

# pca 결과(7) :  0.345104152483673 / (6) 결과 :  0.2736017973179766

# 결과 :  0.28428734040184866
