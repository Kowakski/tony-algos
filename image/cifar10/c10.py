#This is classification for 10 animals:
'''
#About Cifar10 Data
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
There are 50000 training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images.
The test batch contains exactly 1000 randomly-selected images from each class.
The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another.
Between them, the training batches contain exactly 5000 images from each class.
Here are the classes in the dataset, as well as 10 random images from each:
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
'''
import sys
import numpy as np
# from cifar10 import cifar10_input
import cifar10.cifar10_input as input10
import tensorflow as tf
from functools import reduce

Datadir = 'D:\\tmp\\cifar10_data\\cifar-10-batches-bin'

BatchSize = 256
TrainSteps = 10000

images, labels = input10.inputs( False, 'D:\\tmp\\cifar10_data\\cifar-10-batches-bin', BatchSize )

x_input = tf.placeholder( tf.float32, shape=[None, 24, 24, 3] )
y_label = tf.placeholder( tf.float32, shape=[None,10] )

W1 = tf.Variable( initial_value = tf.truncated_normal([5,5,3,64], stddev = 0.001 ), dtype= tf.float32 )
B1 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32 )
tf.summary.histogram("weights1",W1)
tf.summary.histogram("bias1",B1)

x_image = tf.reshape(x_input, [-1,24,24,3])

Net1 = tf.nn.relu( tf.nn.conv2d( x_image, W1, strides=[1,2,2,1], padding='SAME' ) + B1 )
Pool1 = tf.nn.max_pool( Net1, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )

W2 = tf.Variable( initial_value = tf.truncated_normal([3,3,64,64], stddev = 0.001 ), dtype= tf.float32 )
B2 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32 )
tf.summary.histogram("weights2",W1)
tf.summary.histogram("bias2",B1)

Net2 = tf.nn.relu( tf.nn.conv2d( Pool1, W2, strides=[1,2,2,1], padding='SAME' ) + B2 )
Pool2 = tf.nn.max_pool( Net2, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )

nod = 1
for i in Pool2.shape[1:]:
    nod = nod * int(i)

FullayerInput = tf.reshape(Pool2, [-1,nod])

WL1 = tf.Variable( initial_value = tf.truncated_normal([nod,nod*2], stddev = 0.001 ), dtype = tf.float32 )  #weights of logic layer
BL1 = tf.Variable( initial_value = tf.constant(0.1, shape = [1, nod*2]), dtype = tf.float32 )
logits1 = tf.nn.relu( tf.matmul(FullayerInput, WL1 ) + BL1 )

WL = tf.Variable( initial_value = tf.truncated_normal([nod*2,10], stddev = 0.001 ), dtype = tf.float32 )  #weights of logic layer
BL = tf.Variable( initial_value = tf.constant(0.1, shape = [1, 10]), dtype = tf.float32 )
logits = tf.nn.relu( tf.matmul(logits1, WL ) + BL )

softmax = tf.nn.softmax( logits )
EvalLabel =  tf.argmax( softmax, 1 )
evaluation = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(y_label,1), EvalLabel ), tf.float32 ) )

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits( labels = y_label, logits = logits, name = 'loss' ))
tf.summary.scalar('loss',loss)
global_steps = tf.train.get_or_create_global_step()
step = tf.assign_add(global_steps, 1)
learning_rate = tf.train.exponential_decay(0.003,
                                   global_steps,
                                   decay_steps=200,decay_rate=0.98)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize( loss )

sess = tf.Session( )
merged = tf.summary.merge_all( )
writer = tf.summary.FileWriter( "./logs",sess.graph )
sess.run( tf.global_variables_initializer( ) )

tf.train.start_queue_runners(sess = sess)

for i in range(TrainSteps):
    trainx, trainy = sess.run( [ images,labels ] )
    trainy_b = np.eye(10)[trainy]
    sess.run([step,train], feed_dict={ x_input:trainx, y_label:trainy_b })

    if(i%200 == 0):
        evalx, evaly =  sess.run( [ images, labels ] )
        evaly_b = np.eye(10)[evaly]
        print("step:{0}, accuracy:{1}".format( i, sess.run( evaluation, feed_dict={x_input:evalx, y_label:evaly_b} ) ))

    if(i%10 == 0):
        result = sess.run(merged, feed_dict = { x_input:trainx, y_label:trainy_b })
        writer.add_summary(result, i)