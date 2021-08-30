import imp
import tensorflow as tf 
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from tensorflow.python.ops.gen_state_ops import Variable
from tensorflow.python.ops.math_ops import reduce_mean
from tensorflow.python.ops.variable_scope import get_variable

tf.compat.v1.eager_execution()
print(tf.executing_eagerly())
print(tf.__version__)



# tf.set_random_seed(66)


from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.0001
traning_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# 모델구성
 
w1 = tf.compat.v1.get_variable('w1', shape=[3, 3, 1, 32]) # 초기값을 자동으로 넣어줌
                                # [kernel_size, input, output]
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# model = Sequential() 텐서 2 코딩 / 참조
# model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=1, 
#                  padding='same', input_shape=(28, 28, 1),
#                   activation='relu'))
# mode.add(Maxpool2D())

print(L1) # Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
print(L1_maxpool)  # Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 32, 64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(L2)  # Tensor("Selu:0", shape=(?, 14, 14, 64), dtype=float32)
print(L2_maxpool)  # Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

# layer3
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 128])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)  # Tensor("Elu:0", shape=(?, 7, 7, 128), dtype=float32)
print(L3_maxpool)  # Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)

# layer4
w4 = tf.compat.v1.get_variable('w4', shape=[2, 2, 128, 64], )
                        # initializer=tf.contrib.layers.xavier_initializer()) # 가중치 초기화
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1,1,1,1], padding='VALID')
L4 = tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)  # Tensor("LeakyRelu:0", shape=(?, 3, 3, 64), dtype=float32)
print(L4_maxpool)  # Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

# Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*64])
print('Flatten : ', L_flat)

# layer5 DNN
w5 = tf.compat.v1.get_variable('w5', shape=[2*2*64, 64])
b5 = tf.Variable(tf.random_normal([64]), name='b1')
L5 = tf.matmul(L_flat, w5) + b5
L5 = tf.nn.selu(L5)
L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5) # shape=(?, 64)


# layer6 DNN
w6 = tf.compat.v1.get_variable('w6', shape=[64, 32])
b6 = tf.Variable(tf.random_normal([32]), name='b2')
L6 = tf.matmul(L5, w6) + b6
L6 = tf.nn.selu(L6)
L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6) # shape=(?, 32)

# layer7 Softmax
w7 = tf.compat.v1.get_variable('w7', shape=[32, 10])
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

prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1)) # tf.argmax 써야 워닝 안뜸
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('ACC : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test})) # ACC :  0.6814












