#실습
#덧셈
#뺄셈
#곱셈
#나눗셈

import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.div(node1, node2)

print(node6)
# sess = tf.Session()
# print('node1, node2 : ', sess.run([node1, node2]))
# print('sess.run(node3) : ', sess.run(node3))
# print('sess.run(node4) : ', sess.run(node4))
# print('sess.run(node5) : ', sess.run(node5))
# print('sess.run(node6) : ', sess.run(node6))

