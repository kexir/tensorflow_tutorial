import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# flatten this array into a vector of 28x28 = 784 numbers.
# 55,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), 
# and 5,000 points of validation data (mnist.validation). 
# A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension.

# we do a weighted sum of the pixel intensities. The weight is negative 
# if that pixel having a high intensity is evidence against the image being in that class, 
# and positive if it is evidence in favor.

# The result that the evidence for a class given an input is:
# evidence = sum(Wij*xj)+bi 
# where j is number of pixel, x is input, W is weigth, b is bias
# y = software(Wx+b)
import tensorflow as tf
# We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784]
x = tf.placeholder(tf.float32, [None, 784])
# A Variable is a modifiable tensor that lives in TensorFlow's graph of interacting operations. 
# It can be used and even modified by the computation. 
# For machine learning applications, one generally has the model parameters be Variable
# 10 because the images is a number between 0-9
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# First, we multiply x by W with the expression tf.matmul(x, W).
# This is flipped from when we multiplied them in our equation, where we had 
# , as a small trick to deal with x being a 2D tensor with multiple inputs
# x is [55000,784]
# y is [55000,10]
y = tf.nn.softmax(tf.matmul(x, W) + b)
# cross-entropy -sum(y_*log(y))
# cross-entropy is measuring how inefficient our predictions are for describing the truth. 
# To implement cross-entropy we need to first add a new placeholder to input the correct answers:
y_ = tf.placeholder(tf.float32, [None, 10])
# Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1]
# tf.reduce_mean computes the mean over all the examples in the batch.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# In source code, we don't use this formulation, because it is numerically unstable. Instead, 
# we apply tf.nn.softmax_cross_entropy_with_logits on the unnormalized logits 
# (e.g., we call softmax_cross_entropy_with_logits on tf.matmul(x, W) + b),
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# [True, False, True, True] would become [1,0,1,1]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
