import imp
import tensorflow as tf 
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from tensorflow.python.ops.variable_scope import get_variable
tf.set_random_seed(66)

from keras.datasets import mnist, cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)


from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255

learning_rate = 0.0007
traning_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])

# 모델구성
 
w1 = tf.get_variable('w1', shape=[3, 3, 3, 128]) # 초기값을 자동으로 넣어줌
                                # [kernel_size, input, output]
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# model = Sequential() 텐서 2 코딩 / 참조
# model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, 
#                  padding='same', input_shape=(28, 28, 1),
#                   activation='relu'))
# mode.add(Maxpool2D())

print(L1) # shape=(?, 32, 32, 32)
print(L1_maxpool)  # shape=(?, 16, 16, 32)

w2 = tf.get_variable('w2', shape=[3, 3, 128, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L2)  # shape=(?, 16, 16, 64)
print(L2_maxpool)  # shape=(?, 8, 8, 64)

# layer3
w3 = tf.get_variable('w3', shape=[3, 3, 64, 64])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)  # shape=(?, 8, 8, 128)
print(L3_maxpool)  #  shape=(?, 4, 4, 128)

# layer4
w4 = tf.get_variable('w4', shape=[2, 2, 64, 64], 
                        initializer=tf.contrib.layers.xavier_initializer()) # 가중치 초기화
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1,1,1,1], padding='VALID')
L4 = tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)  # shape=(?, 3, 3, 64)
print(L4_maxpool)  # shape=(?, 2, 2, 64)

# Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*64])
print('Flatten : ', L_flat) # shape=(?, 256)

# layer5 DNN
w5 = get_variable('w5', shape=[2*2*64, 64])
b5 = tf.Variable(tf.random_normal([64]), name='b1')
L5 = tf.matmul(L_flat, w5) + b5
L5 = tf.nn.selu(L5)
L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5) # shape=(?, 64)


# layer6 DNN
w6 = get_variable('w6', shape=[64, 32])
b6 = tf.Variable(tf.random_normal([32]), name='b2')
L6 = tf.matmul(L5, w6) + b6
L6 = tf.nn.selu(L6)
L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6) # shape=(?, 32)

# layer7 Softmax
w7 = get_variable('w7', shape=[32, 10])
b7 = tf.Variable(tf.random_normal([10]), name='b3')
L7 = tf.matmul(L6, w7) + b7
hypothesis = tf.nn.softmax(L7)
print(L7)  # shape=(?, 10)

# 3. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1)) # categorical_crossentropy
# cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary crossentropy
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# learning_rate = 0.001
# traning_epochs = 15
# batch_size = 100
# total_batch = int(len(x_train)/batch_size)

for epoch in range(traning_epochs):
    avg_loss = 0

    for i in range(total_batch):  # 600번 돈다
        start = i *batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x:batch_x, y:batch_y}

        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_loss += batch_loss/total_batch

    print('Epoch : ', '%04d' %(epoch + 1), 'loss : {:.9f}'.format(avg_loss))

prediction = tf.equal(tf.math.argmax(hypothesis, 1), tf.argmax(y, 1)) # tf.argmax 써야 워닝 안뜸
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('ACC : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test})) 

'''
ACC :  0.2674 / lr: 0.0001
ACC :  0.4222 / lr: 0.0005
ACC :  0.4262 / lr: 0.0007
ACC :  0.4465 / 모델 변경
'''












