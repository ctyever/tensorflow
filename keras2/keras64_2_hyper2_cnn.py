# 실습 
# CNN 으로 변경
# 파라미터 변경
# 노드의 갯수, activation 도 추가
# learnig_rate 추가

import numpy as np
from tensorflow.keras.datasets import fashion_mnist, cifar100, mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

# 2. 모델
def build_model(drop, opt, lr, node1, node2, activation):
    inputs = Input(shape=(28, 28, 1), name='input')
    x = Conv2D(node1, (2,2), activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(node2, (2,2), activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Conv2D(128, (2,2), activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=opt(learning_rate=lr), metrics=['acc'],
                    loss='categorical_crossentropy')
    return model

from tensorflow.keras.optimizers import Adam, Adadelta

# optimizer = Adam(lr = 0.001)

def create_hyperparameter():
    batches = [100, 200, 300, 400, 500]
    optimizer = [Adam, Adadelta]
    lr = [0.1, 0.2, 0.3]
    dropout=[0.3,0.4,0.5]
    node1=[512,256]
    node2=[256,128]
    activation=['relu', 'tanh']
    epochs=[1,2]
    return {'batch_size' : batches, 'opt': optimizer, 'lr' :lr,
            'drop' : dropout, 'node1' : node1, 'node2' : node2, 'activation' : activation, 'epochs': epochs}

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

# {'activation': 'tanh', 'batch_size': 10, 'drop': 0.3, 'epochs': 2, 'node1': 64, 'node2': 16, 'optimizer': 'rmsprop'}
# 최종 스코어 :  0.6374269127845764

