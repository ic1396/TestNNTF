#!/usr/bin/python3
# 《TensorFlow神经网络编程》第 1 章
# 第一节  解线性方程

import tensorflow as tf

# equation 1
x1 = tf.constant(3, dtype=tf.float32)
y1 = tf.constant(2, dtype=tf.float32)
point1 = tf.stack([x1, y1])

# equation 2
x2 = tf.constant(4, dtype=tf.float32)
y2 = tf.constant(-1, dtype=tf.float32)
point2 = tf.stack([x2, y2])

# solve for AX = C
X = tf.transpose(tf.stack([point1, point2]))
C = tf.ones((1, 2), dtype=tf.float32)

A = tf.matmul(C, tf.matrix_inverse(X))

with tf.Session() as sess:
    X = sess.run(X)
    print(X)
    
    A = sess.run(A)
    print(A)

b = 1 / A[0][1]
a = -b + A[0][0]

print("Hence Linear Equation is y = {a}x + {b}". format(a=a, b=b))