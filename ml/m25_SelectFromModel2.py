# 실습
# 1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 r2값과 featureimportance 구할 것

# 2. 위 스레드값으로 SelectFromModel 돌려서 최적의 피처 갯수 구할 것

# 3. 위 피처 갯수로 피쳐갯수를 조정한뒤
# 그걸로 다시 랜덤서치 그리드서치해서
# 최적의 r2 구할 것

# 1번값과 3번값 비교

from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import SelectFromModel



# 1. 데이터 구성

datasets = load_diabetes()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)


# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# 2. 모델
# 2. 모델 구성

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100, 200]},
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]

# model = GridSearchCV(XGBRegressor(), parameters, cv=kfold, verbose=1)

model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.300000012, max_delta_step=0, max_depth=6,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric='rmse', # 'mae', 'logloss'
            #eval_set=[(x_train, y_train), (x_test, y_test)]
)

# 4. 평가
# print('최적의 매개변수 : ', model.best_estimator_)

print('model.score', model.score(x_test, y_test))

thresholds = np.sort(model.feature_importances_)

# model.score 0.44221928509211916
# r2_score :  0.44221928509211916


for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # print(selection)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1], score*100))


