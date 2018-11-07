import numpy as np
import tensorflow as tf
import time 

# OPERATIONS WITH CONSTANTS
x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
	print("\nOperations with Constants\n")
	print("Addition: {}".format(sess.run(x+y)))
	print("Subtraction: {}".format(sess.run(x-y)))
	print("Multiplication: {}".format(sess.run(x*y)))
	print("Division: {}".format(sess.run(x/y)))

# OPERATIONS WITH PLACEHOLDERS
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

add = tf.add(x,y)
sub = tf.subtract(x,y)
mul = tf.multiply(x,y)
div = tf.divide(x,y)

# fd = {x:20,y:30}
# fd = {x:[20,30],y:[30,40]}
fd = {x:np.arange(1,100000000),y:np.arange(-100000000,-1)}

with tf.Session() as sess:
	print("\nOperations with Placeholders\n")
	print("Addition: {}".format(sess.run(add,feed_dict=fd)))
	print("Subtraction: {}".format(sess.run(sub,feed_dict=fd)))
	print("Multiplication: {}".format(sess.run(mul,feed_dict=fd)))
	print("Division: {}".format(sess.run(div,feed_dict=fd)))

# MATRIX OPERATIONS
# a = np.array([5.0,5.0]).reshape(2,1)
# b = np.array([2.0,2.0]).reshape(1,2)
a = np.arange(1,10001,dtype=np.int32).reshape(2500,4)
b = np.arange(1,10001,dtype=np.int32).reshape(4,2500)

mat1 = tf.constant(a)
mat2 = tf.constant(b)

mm = tf.matmul(mat1,mat2)

with tf.Session() as sess:
	result = sess.run(mm)


print(result)

sess.close()