import tensorflow as tf
tf.set_random_seed(66)

# 1. 데이터 구성
x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]] # (6,2)
y_data = [[0], [0], [0], [1], [1], [1]] # (6,1)

#2. 모델
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2, 1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))

# hypothesis = x * w + b
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis-y)) # mse
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary crossentropy

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 3. 훈련
for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
              feed_dict={x:x_data, y:y_data})  # 이부분 수정할 것
    if epochs % 200 == 0:
        print(epochs, "cost : ", cost_val, "\n", hy_val)

# 4. 평가 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))


print("===========================================================")

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})
print('예측값 : ', hy_val, '\n 예측 결과값 : ', c, '\n Accuracy : ', a)

sess.close()
