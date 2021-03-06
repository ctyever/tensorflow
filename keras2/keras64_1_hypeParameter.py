import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D
from tensorflow.python.keras.backend import dropout

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255

# 2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                    loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [1000, 2000, 3000, 4000, 5000]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    dropout=[0.3,0.4,0.5]
    return {'batch_size' : batches, 'optimizer': optimizer,
            'drop' : dropout}

hyperparameters = create_hyperparameter()
# print(hyperparameters)


###############################################################################################################################
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor  # 사이킷런으로 케라스(텐서플로우)모델 감쌈
model2 = KerasClassifier(build_fn=build_model, verbose=1, validation_split=0.2) #, epochs=2)  # ***epochs는 2군데 먹힘-1, validation_split도 2군데 먹힘
###############################################################################################################################

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
model = GridSearchCV(model2, hyperparameters, cv=5)

model.fit(x_train, y_train, verbose=1, epochs=3, validation_split=0.2)

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_test, y_test)
print("최종 스코어 : ", acc)

