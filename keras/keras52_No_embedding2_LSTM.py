from numpy.matrixlib.defmatrix import matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import numpy as np  

# 1. 데이터
docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글세요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밋네요', '청순이가 잘 생기긴 했어요'
]

# 긍정을 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
# print(token.word_index)

x = token.texts_to_sequences(docs)
# print(x)
# [[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
# 크기를 맞춰주기 위해 0을 채움, padding, 0을 뒤에 채워도 되지만 앞에 채움, lstm 영향치 때문

from tensorflow.keras.preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre', maxlen=5) # maxlen 을 더 작게 할 수 있음, 데이터가 많아지면 maxlen 을 줄일 필요가 생김, 
# padding='post' 하면 뒤에 0이 채워짐
# print(pad_x)
# '''
# [[ 0  0  0  2  4]
#  [ 0  0  0  1  5]
#  [ 0  1  3  6  7]
#  [ 0  0  8  9 10]
#  [11 12 13 14 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0  0 17]
#  [ 0  0  0 18 19]
#  [ 0  0  0 20 21]
#  [ 0  0  0  0 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0 25  3 26 27]]
# '''
# print(pad_x.shape) # (13, 5)
# 원핫코딩 하면 뭐로 바껴? (13, 5, 28)
# 옥스포드? (13, 5, 10000000)

pad_x = pad_x.reshape(13, 5, 1)

word_size = len(token.word_index)
# print(word_size) # 27

# print(np.unique(pad_x))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27]


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
                 # 단어사전의 개수                   단어수, 길이
# model.add(Embedding(input_dim=28, output_dim=11, input_length=5))
# input_length 안 쒀줘도 되는데 자동으로 인식하면서 None 으로 인식함
# model.add(Embedding(128, 77)) # input_dim 이 단어개수 보다 많으면 됨, 그런데 맞춰주는게 좋음
model.add(LSTM(32, input_shape=(5,1)))
model.add(Dense(1, activation='sigmoid'))

model.summary()
'''

'''

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=32)

# 4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)
