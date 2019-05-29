#!/usr/bin/python3
# 《TensorFlow神经网络编程》第 1 章
# 第一节  主成分分析（PCA）

# 在文本数据上使用 TensorFlow SVD 运算

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as FF
import pandas as pd

path = "./"   # 使用当前文件所在目录
logs = "./"   # 使用当前文件所在目录

xMatrix = np.array([[0, 2, 1, 0, 0, 0, 0, 0],
                    [2, 0, 0, 1, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 1, 1, 0]], dtype=np.float32)

def pca(mat):
    mat = tf.constant(mat, dtype=tf.float32)
    mean = tf.reduce_mean(mat, 0)
    less = mat - mean
    s, u, v = tf.svd(less, full_matrices=True, compute_uv=True)
    s2 = s ** 2
    variance_ratio = s2 / tf.reduce_sum(s2)
    
    with tf.Session() as sess:
        run = sess.run([variance_ratio])
    return run

if __name__ == '__main__':
    print(pca(xMatrix))