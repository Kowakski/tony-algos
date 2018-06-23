import sys
import time
import numpy as np
import tensorflow as tf

import cifar10.cifar10_input as input10

sys.path.append("../")     #set path import cifar10 codes

Datadir = '/tmp/cifar10_data/cifar-10-batches-bin'

BatchSize = 256
TrainSteps = 10000
lr = 0.003

images, labels = input10.inputs( False, data_dir = Datadir, batch_size = BatchSize )
images_eval, labels_eval = input10.inputs( True, data_dir = Datadir, batch_size = BatchSize )

x_input = tf.placeholder( tf.float32, shape=[None, 24, 24, 3] )
y_label = tf.placeholder( tf.float32, shape=[None,10] )

with tf.Session() as sess:
    net = vgg19( pre_mod_path )
    net.build()    #construct the net
    sess.run(net.prob, feed_dict={ x_input:images } )
    loss = tf.sum_reduce((y_label-net.prob)**2)
    optimizer = tf.train.MomentumOptimizer(lr)
    while i in range(TrainSteps):
        optimizer.minimize(loss)
        if i%100 ==0:
            print("step:{0}, loss:{1}".format(i, loss))