import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import r2_score, accuracy_score
from sklearn.decomposition import PCA
import warnings 
warnings.filterwarnings(action='ignore')


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x = np.append(x_train, x_test, axis=0)
# # print(x.shape) # (70000, 28, 28)
x = x.reshape(70000, 28*28)


pca = PCA(n_components=154) # 
x = pca.fit_transform(x)
# print(x.shape) # 

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)

# cumsum = np.cumsum(pca_EVR)
# print(cumsum)

# print(np.argmax(cumsum >= 0.999)+1) # 0.95 / 154, 0.99 / 331, 0.999 / 486

x_train = x[:60000]
x_test = x[60000:]


# print(y_train.shape, y_test.shape)

# print(x_train.shape)
# print(x_test.shape)

# print(np.unique(y_train))

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer


# print(x_train)

# 2. 모델 구성
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

pipe = Pipeline([("scaler", MinMaxScaler()), ("xg", XGBClassifier())]) # 


params = [
    {"xg__n_estimators":[100, 200, 300], 
    "xg__learning_rate":[0.001, 0.01],
    "xg__max_depth":[4, 5, 6], 
    "xg__colsample_bytree":[0.6, 0.9, 1], 
    "xg__colsample_bylevel":[0.6, 0.7, 0.9],
    }
]

model = GridSearchCV(pipe, params, cv=kfold, verbose=1)

# model.summary()

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('최적의 매개변수 : ', model.best_estimator_)
print('best_params_ : ', model.best_params_)
print('best_score_ : ', model.best_score_)

print('model.score', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))

# best_params_ :  {'xg__eta': 0.1, 'xg__eval_metric': 'logloss', 'xg__max_depth': 3, 'xg__objective': 'binary:logistic'}
# best_score_ :  0.9327833333333334
# model.score 0.9368
# accuracy_score :  0.9368

#pca 154
# best_params_ :  {'xg__eta': 0.1, 'xg__eval_metric': 'logloss', 'xg__max_depth': 3, 'xg__objective': 'binary:logistic'}
# best_score_ :  0.9076000000000001
# model.score 0.9138
# accuracy_score :  0.9138

