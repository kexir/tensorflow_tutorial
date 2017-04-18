import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

W = tf.Variable([-1.], tf.float32)
b = tf.Variable([1.], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
