#!/usr/bin/python3
# 《TensorFlow神经网络编程》第 1 章
# 第一节  解圆的正则方程

# canonical circle equation
# x**2 +2)  ==> AX = B
# we have to solve for d, e, f

import tensorflow as tf

points = tf.constant([[2, 1], [0, 5], [-1, 2]], dtype=tf.float64)
X = tf.constant([[2, 1, 1], [0, 5, 1], [-1, 2, 1]], dtype=tf.float64)
B = -tf.constant([[5], [25], [15]], dtype=tf.float64)

A = tf.matrix_solve(X, B)

with tf.Session() as sess:
    result = sess.run(A)
    D, E, F = result.flatten()
    print("Hence Circle Equation is: x**2 + y**2 +{D}x + {E}y + {F} = 0".format(**locals()))