# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:21:20 2017

@author: Administrator
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  #导入MNIST数据集


mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
print(mnist)


#从MNIST数据集中筛选出5000条数据用作测试
train_X,train_Y = mnist.train.next_batch(5000)
#从MNIST数据集中筛选出200条数据用作测试
test_X,test_Y = mnist.test.next_batch(100)

#图输入
train2_X = tf.placeholder("float",[None,784])   #输入batchsize数目的图片
test2_X = tf.placeholder("float",[784])         #输入了一个图片，这个图片用来预测的

#使用L1距离计算KNN距离计算   L1是曼哈顿距离，就是坐标相减的绝对值相加
distance = tf.reduce_sum(tf.abs(tf.add(train2_X,tf.negative(test2_X))),reduction_indices=1)       #计算和batchsize个图片里面所有图片的距离

#预测：取得最近的邻居节点
pred = tf.arg_min(distance,0)    #这里取K=1，取到这个索引值，然后到trainlabel里面把标签取出来就好了

accuracy = 0

#变量初始化
init = tf.global_variables_initializer()

#启动图
with tf.Session() as sess:
    sess.run(init)
    #遍历测试数据集
    for i in range(len(test_X)):
        #获取最近的邻居节点
        nn_index = sess.run(pred,feed_dict={train2_X:train_X,test2_X:test_X[i,:]})
        #获取最近的邻居节点的类别标签，并将其与该节点的真实类别标签进行比较
        print("测试数据",i,"预测分类:",np.argmax(train_Y[nn_index]),"真实类别:",np.argmax(test_Y[i]))
        #计算准确率
        if np.argmax(train_Y[nn_index]) == np.argmax(test_Y[i]):
            accuracy += 1./len(test_X)
    print("分类准确率为:",accuracy)