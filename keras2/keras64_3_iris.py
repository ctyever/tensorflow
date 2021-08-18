import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.python.keras.backend import dropout
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings(action='ignore') 


# 1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.7, shuffle=True, random_state=9)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# 2. 모델
def build_model(drop=0.5, optimizer='adam', node1=32, node2=16, activation='relu'):
    inputs = Input(shape=(4,), name='input')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                    loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [10, 20, 30, 40, 50]
    # optimizer = ['rmsprop', 'adam', 'adadelta']
    # dropout=[0.3,0.4,0.5]
    # node1=[64,32]
    # node2=[16,8]
    # activation=['relu', 'tanh']
    # epochs=[1,2]
    return {'batch_size' : batches,} # 'optimizer': optimizer,
            # 'drop' : dropout, 'node1' : node1, 'node2' : node2, 'activation' : activation, 'epochs': epochs}

hyperparameters = create_hyperparameter()
# print(hyperparameters)


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# model2 = build_model()
model2 = KerasClassifier(build_fn=build_model, verbose=1) # , epochs=2, validation_split=0.2) # 한번 랩핑한다

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
model = GridSearchCV(model2, hyperparameters, cv=5)

model.fit(x_train, y_train, verbose=1, epochs=1, validation_split=0.2)

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_test, y_test)
print("최종 스코어 : ", acc)

