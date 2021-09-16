import tensorflow as tf 

# print(tf.__version__)

# print('hello world')

hello = tf.constant("Hello World")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session() # warning : Please use tf.compat.v1.Session instead.
sess = tf.compat.v1.Session()
print(sess.run(hello))
# b'Hello World'