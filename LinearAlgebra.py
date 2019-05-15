#!/usr/bin/python3
# 《TensorFlow神经网络编程》第 1 章
# 第一节

# 向量 Vectors
"""
import tensorflow as tf

vector = tf.constant([[4, 5, 6]], dtype=tf.float32)
eucNorm = tf.norm(vector, ord="euclidean")
with tf.Session() as sess:
    print(sess.run(eucNorm))
"""

# 矩阵 Matrix
'''
# 将不同的矩阵转换为张量对象
# convert matrices to tensor objects
import numpy as np
import tensorflow as tf

#create a 2 * 2 matrix in various forms
matrix1 = [[1.0, 2.0], [3.0, 4.0]]
matrix2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
matrix3 = tf.constant([[1.0, 2.0], [3.0, 4.0]])

print(matrix1)
print(matrix2)
print(matrix3)

print(type(matrix1))
print(type(matrix2))
print(type(matrix3))

tensorForM1 = tf.convert_to_tensor(matrix1, dtype=tf.float32)
tensorForM2 = tf.convert_to_tensor(matrix2, dtype=tf.float32)
tensorForM3 = tf.convert_to_tensor(matrix3, dtype=tf.float32)

print(tensorForM1)
print(tensorForM2)
print(tensorForM3)

print(type(tensorForM1))
print(type(tensorForM2))
print(type(tensorForM3))
'''

# 矩阵乘法
'''
# 哈达玛积（Hadamard product）和点积（dot product）
import tensorflow as tf
mat1 = tf.constant([[4, 5, 6], [3, 2, 1]])
mat2 = tf.constant([[7, 8, 9], [10, 11, 12]])

# Hadamard product (element wise)
mult = tf.multiply(mat1, mat2)

# dot product (no. of rows = no. of cloumns)
dotprod = tf.matmul(mat1, tf.transpose(mat2))

with tf.Session() as sess:
    print(sess.run(mult))
    print(sess.run(dotprod))
'''

# 迹运算 Trace
'''
import tensorflow as tf
mat = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.float32)

# get trace ('sum of diagonal elements') of the matrix
mat = tf.trace(mat)

with tf.Session() as sess:
    print(sess.run(mat))
'''

# 矩阵转置
'''
import tensorflow as tf

x = [[1, 2, 3], [4, 5, 6]]
x = tf.convert_to_tensor(x)
xtrans = tf.transpose(x)

y = ([[[1, 2, 3], [6, 5, 4]], [[4, 5, 6], [3, 6, 3]]])
y = tf.convert_to_tensor(y)
ytrans = tf.transpose(y, perm=[0, 2, 1])

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(xtrans))
    print(sess.run(y))
    print(sess.run(ytrans))
'''

# 对角矩阵
'''
import tensorflow as tf
mat = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.float32)

# get diagonal of the matrix
diag_mat = tf.diag_part(mat)

# create matrix with given diagonal
mat = tf.diag([1, 2, 3, 4])

with tf.Session() as sess:
    print(sess.run(diag_mat))
    print(sess.run(mat))
'''

# 单位矩阵
'''
import tensorflow as tf
identity = tf.eye(3, 3)
with tf.Session() as sess:
    print(sess.run(identity))
'''

# 逆矩阵
'''
import tensorflow as tf

mat = tf.constant([[2, 3, 4], [5, 6, 7], [8, 9, 10]], dtype=tf.float32)
# mat = tf.constant([[0, 1, 2], [1, 1, -1], [2, 4, 0]], dtype=tf.float32)
# mat = tf.constant([[0, 1, 2], [1, 1, 4], [2, -1, 0]], dtype=tf.float32)
print(mat)

inv_mat = tf.matrix_inverse(tf.transpose(mat))

with tf.Session() as sess:
    print(sess.run(inv_mat))
'''