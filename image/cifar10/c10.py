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
0 airplane
1 automobile
2 bird
3 cat
4 deer
5 dog
6 frog
7 horse
8 ship
9 truck
'''
import sys
import numpy as np
# from cifar10 import cifar10_input
import cifar10.cifar10_input as input10
import tensorflow as tf
from functools import reduce
from tensorflow.contrib.layers.python.layers import batch_norm

Datadir = '/tmp/cifar10_data/cifar-10-batches-bin'

BatchSize = 256
TrainSteps = 18000

images, labels = input10.inputs( False, data_dir = Datadir, batch_size = BatchSize )
images_eval, labels_eval = input10.inputs( True, data_dir = Datadir, batch_size = BatchSize )

x_input = tf.placeholder( tf.float32, shape=[None, 24, 24, 3] )
y_label = tf.placeholder( tf.float32, shape=[None,10] )
trainflag = tf.placeholder(tf.bool)

'''
x_image = tf.reshape(x_input, [-1,24,24,3])

Net1 = tf.layers.conv2d( x_image, filters = 64, kernel_size = [5,5], strides = (1,1), padding = 'same', name = 'net1')
Pool1 = tf.layers.max_pooling2d(Net1, pool_size=[3,3], strides=[2,2], padding='same', name='pool1')

Net2 = tf.layers.conv2d( Pool1, filters = 64, kernel_size = [3,3], strides = (1,1), padding = 'same', name = 'net2')
Pool2 = tf.layers.max_pooling2d(Net2, pool_size=[3,3], strides=[2,2], padding='same', name='pool2')

Net3 = tf.layers.conv2d( Pool1, filters = 64, kernel_size = [3,3], strides = (1,1), padding = 'same', name = 'net3')
Pool3 = tf.layers.max_pooling2d(Net2, pool_size=[3,3], strides=[2,2], padding='same', name='pool3')

nod = 1
for i in Pool3.shape[1:]:
    nod = nod * int(i)

FullayerInput = tf.reshape(Pool3, [-1,nod])

logits = tf.contrib.layers.fully_connected(FullayerInput, 10)
'''

def batch_norm_layer(value,train = None, name = 'batch_norm'):
  if train is not None:
      return batch_norm(value, decay = 0.9,updates_collections=None, is_training = True)
  else:
      return batch_norm(value, decay = 0.9,updates_collections=None, is_training = False)

#layer1
x_image = tf.reshape(x_input, [-1,24,24,3])

W1 = tf.Variable( initial_value = tf.truncated_normal([5,1,3,64], stddev = 0.001 ), dtype= tf.float32, name = 'W1' )
B1 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32 , name = 'B1')
W1r = tf.Variable( initial_value = tf.truncated_normal([1,5,64,64], stddev = 0.001 ), dtype= tf.float32, name = 'W1' )
B1r = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32 , name = 'B1')

Net1r = tf.nn.relu( batch_norm_layer((tf.nn.conv2d( x_image, W1, strides=[1,1,1,1], padding='SAME' ) + B1), trainflag ))
Pool1r = tf.nn.max_pool( Net1r, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )

Net1 = tf.nn.relu( batch_norm_layer((tf.nn.conv2d( Pool1r, W1r, strides=[1,1,1,1], padding='SAME' ) + B1r),trainflag)) #batch normalization 是针对激活函数的
Pool1 = tf.nn.max_pool( Net1, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )

#layer2, 多通道卷积
W2_1 = tf.Variable( initial_value = tf.truncated_normal([1,1,64,64], stddev = 0.001 ), dtype= tf.float32, name = 'W2' )
B2_1 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32, name = 'B2' )
Net2_1 = tf.nn.relu(batch_norm_layer(( tf.nn.conv2d( Pool1, W2_1, strides=[1,1,1,1], padding='SAME' ) + B2_1 ), trainflag))
# Pool2_1 = tf.nn.max_pool( Net2, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )

W2_3 = tf.Variable( initial_value = tf.truncated_normal([3,3,64,64], stddev = 0.001 ), dtype= tf.float32, name = 'W2' )
B2_3 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32, name = 'B2' )
Net2_3 = tf.nn.relu(batch_norm_layer(( tf.nn.conv2d( Pool1, W2_3, strides=[1,1,1,1], padding='SAME' ) + B2_3 ), trainflag))

W2_5 = tf.Variable( initial_value = tf.truncated_normal([5,5,64,64], stddev = 0.001 ), dtype= tf.float32, name = 'W2' )
B2_5 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32, name = 'B2' )
Net2_5 = tf.nn.relu(batch_norm_layer(( tf.nn.conv2d( Pool1, W2_5, strides=[1,1,1,1], padding='SAME' ) + B2_5 ), trainflag))

Net2 = tf.concat([Net2_1, Net2_3, Net2_5],3)

Pool2 = tf.nn.max_pool( Net2, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )

#layer3
W3 = tf.Variable( initial_value = tf.truncated_normal([3,3,192,64], stddev = 0.001 ), dtype= tf.float32, name = 'W3' )
B3 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32, name = 'B3' )
Net3 = tf.nn.relu( tf.nn.conv2d( Pool2, W3, strides=[1,1,1,1], padding='SAME' ) + B3 )
Pool3 = tf.nn.max_pool( Net2, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )

print( "Pool2 is: ", Pool3 )
nod = 1
for i in Pool3.shape[1:]:
    nod = nod * int(i)

FullayerInput = tf.reshape(Pool3, [-1,nod])

WL = tf.Variable( initial_value = tf.truncated_normal([nod,10], stddev = 0.001 ), dtype = tf.float32, name = 'WL' )  #weights of logic layer
BL = tf.Variable( initial_value = tf.constant(0.1, shape = [1, 10]), dtype = tf.float32, name = 'BL' )
logits = tf.nn.relu( tf.matmul(FullayerInput, WL ) + BL )

softmax = tf.nn.softmax( logits )
EvalLabel =  tf.argmax( softmax, 1 )
evaluation = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(y_label,1), EvalLabel ), tf.float32 ) )

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits( labels = y_label, logits = logits, name = 'loss' ))
tf.summary.scalar('loss',loss)
global_steps = tf.train.get_or_create_global_step()
step = tf.assign_add(global_steps, 1)
learning_rate = tf.train.exponential_decay(2e-4,
                                   global_steps,
                                   decay_steps=500,decay_rate=0.95)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize( loss )

sess = tf.Session( )
merged = tf.summary.merge_all( )
writer = tf.summary.FileWriter( "./logs",sess.graph )
sess.run( tf.global_variables_initializer( ) )
saver = tf.train.Saver()
tf.train.start_queue_runners(sess = sess)

for i in range(TrainSteps):
    trainx, trainy = sess.run( [ images,labels ] )
    trainy_b = np.eye(10)[trainy]
    sess.run([step,train], feed_dict={ x_input:trainx, y_label:trainy_b, trainflag:1 })

    if(i%200 == 0):
        evalx, evaly =  sess.run( [ images_eval, labels_eval ] )
        evaly_b = np.eye(10)[evaly]
        print("step:{0}, accuracy:{1}".format( i, sess.run( evaluation, feed_dict={x_input:evalx, y_label:evaly_b } ) ))

    # if(i%10 == 0):
    #     result = sess.run(merged, feed_dict = { x_input:trainx, y_label:trainy_b })
    #     writer.add_summary(result, i)
saver.save(sess, "my_net/train_result.ckpt")
print("save train result")
