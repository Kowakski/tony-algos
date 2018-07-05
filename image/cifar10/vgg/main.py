#use floyd run --data kowalski8787/datasets/cifar10_py/1:/slnmount "python main.py"
import sys
import time
import vgg19
from PIL import Image
import numpy as np
import tensorflow as tf
import pdb
from tensorflow.python import debug as tf_debug
from cifar10_input import C10Input

BatchSize = 2
TrainSteps = 10000
lr = 0.01
pre_mod_path = None

# c10input = C10Input('/slnmount/')
c10input = C10Input('/tmp/cifar-10-batches-py')

x_input = tf.placeholder( tf.float32, shape=[None, 32, 32, 3] )
y_label = tf.placeholder( tf.float32, shape=[None,10] )
train_mode = tf.placeholder( tf.bool )

#Resized Image
Rimages = tf.image.resize_images( x_input, [224, 224], method=tf.image.ResizeMethod.BILINEAR, align_corners=False )

sess = tf.Session( )

net = vgg19.Vgg19( pre_mod_path )

net.build( Rimages, train_mode )    #construct the net
print(net.get_var_count())

sess.run( tf.global_variables_initializer( ) )

loss = tf.reduce_sum((net.prob - y_label)**2)
print("loss shape: {0}".format(loss))
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss)

images, labels = c10input.get_batch_data( BatchSize )
images = images/255.0
trainy_b = np.eye(10)[labels]

for i in range(TrainSteps):
    sess.run( train, feed_dict={ x_input:images, y_label:trainy_b, train_mode:True } )
    if i%50 ==0:
        print("step:{0}, loss:{1}".format(i, sess.run(loss, feed_dict={ x_input:images, y_label:trainy_b, train_mode:False })))