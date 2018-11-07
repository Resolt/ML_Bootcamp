import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# PART 1 - READING AND EXPLORING MNIST DATA

# LINK MNIST
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

print(mnist.train.images.shape)

print(mnist.train.images[1].shape)

# plt.imshow(mnist.train.images[1].reshape(28,28),cmap='gist_gray')
# plt.show()

ex = mnist.train.images[1].reshape(mnist.train.images[1].shape[0],1)
print(ex)

# sns.heatmap(ex)
# plt.show()

# PART 2 -

x = tf.placeholder(dtype=tf.float32,shape=[None,784]) # PLACE HOLDER FOR THE INPUT - WE KNOW ITS 784, BUT WE HAVEN'T DECIDED ON BATCH SIZE
W = tf.Variable(tf.zeros([784,10])) # THE WEIGHTS - 784 FOR THE PIXELS - 10 FOR EACH POSSIBLE VALUE (WE ARE LOOKING TO DETERMINE A NUMBER BETWEEN 0 and 9)
b = tf.Variable(tf.zeros([10])) # BIASES

y = tf.matmul(x,W) + b # THE OUTPUT

y_true = tf.placeholder(tf.float32,shape=[None,10]) # THIS IS THE SAME AS y_train. WE DON'T KNOW THE BATCH SIZE JUST YET BUT WE DO KNOW THE POSSIBLE OUTPUTS

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	
	for step in range(1000):
		batch_x,batch_y = mnist.train.next_batch(100)
		sess.run(train,feed_dict={x:batch_x,y_true:batch_y})

	matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))

	acc = tf.reduce_mean(tf.cast(matches,tf.float32))

	print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))