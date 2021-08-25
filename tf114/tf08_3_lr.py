# 실습
# tf08_2 파일의 lr을 수정해서
# epoch가 2000번이 아니라 100 번 이하로 줄여라
# 결과치는
# step=100 이하, w=1.9999, b=0.9999

import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import shape
tf.set_random_seed(66)

# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])
x_test = tf.placeholder(tf.float32, shape=[None])


# W = tf.Variable([1], dtype=tf.float32 ) # 랜덤하게 내맘대로 넣어준 초기값
# b = tf.Variable(1, dtype=tf.float32 ) # 랜덤하게 내맘대로 넣어준 초기값

W = tf.Variable(tf.random_normal([1]), dtype = tf.float32) # 랜덤하게 내맘대로 넣어준
b = tf.Variable(tf.random_normal([1]), dtype = tf.float32) 

hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.17338)
train = optimizer.minimize(loss)

predict = x_test * W + b



sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    # sess.run(train)
    _, loss_val, W_val, b_val, predict_val =  sess.run([train, loss, W, b, predict], 
                                        feed_dict={x_train:[1,2,3], y_train:[3,5,7], x_test:[5, 6]})
    if step % 20 == 0:
        # print(step, sess.run(loss), sess.run(W), sess.run(b))
        print(step, loss_val, W_val, b_val, predict_val)

'''
0 24.18615 [3.659217] [2.015533] [0.30430025 0.28335133]
20 0.99476177 [2.312687] [1.2594604] [ 8.765111  10.3308935]
40 0.04109185 [2.0533035] [1.0754781] [10.518392 12.419878]
60 0.0017298517 [2.0065055] [1.0250677] [10.890161 12.865709]
80 7.869607e-05 [1.9994844] [1.0092504] [10.972517 12.965648]
'''