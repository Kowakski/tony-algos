import sys
import time
import vgg19
from PIL import Image
import numpy as np
import tensorflow as tf
import pdb
from tensorflow.python import debug as tf_debug

sys.path.append("../")     #set path import cifar10 codes

import cifar10.cifar10_input as input10
Datadir = '/tmp/cifar10_data/cifar-10-batches-bin'

BatchSize = 128
TrainSteps = 10000
lr = 0.3
pre_mod_path = None
# pre_mod_path='./vgg19.npy'

images, labels = input10.inputs( False, data_dir = Datadir, batch_size = BatchSize )
Rimages = tf.image.resize_images( images, [224, 224], method=tf.image.ResizeMethod.BILINEAR, align_corners=False )
print("Rimage size is: ", np.shape(Rimages))
images_eval, labels_eval = input10.inputs( True, data_dir = Datadir, batch_size = BatchSize )

x_input = tf.placeholder( tf.float32, shape=[None, 224, 224, 3] )
y_label = tf.placeholder( tf.float32, shape=[None,10] )
train_mode = tf.placeholder( tf.bool )


sess = tf.Session( )

net = vgg19.Vgg19( pre_mod_path )

net.build( x_input, train_mode )    #construct the net
sess.run( tf.global_variables_initializer( ) )
tf.train.start_queue_runners(sess = sess)
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
loss = tf.reduce_sum( (y_label-net.prob)**2)
optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(loss)

for i in range(TrainSteps):
    trainx, trainy = sess.run([Rimages, labels])
    trainy_b = np.eye(10)[trainy]
    # pdb.set_trace()
    sess.run( train, feed_dict={ x_input:trainx, y_label:trainy_b, train_mode:True } )
    if i%50 ==0:
        print("step:{0}, loss:{1}".format(i, sess.run(loss, feed_dict={ x_input:trainx, y_label:trainy_b, train_mode:False })))