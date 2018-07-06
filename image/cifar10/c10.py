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
import time
import numpy as np
from cifar10_input import C10Input
import cifar10.cifar10_input as input10
import tensorflow as tf
from functools import reduce
from tensorflow.contrib.layers.python.layers import batch_norm

Datadir = '/tmp/cifar-10-batches-py'
# Datadir = '/tmp/cifar10_data/cifar-10-batches-bin'
# Datadir = '/home/shenlin/share/Traindata/cifar10_data/cifar-10-batches-bin'
# Datadir = '/media/shenlin/slnmobile/DatasSet/cifar-10-python.tar.gz'

BatchSize = 256
TrainSteps = 10000

# images, labels = input10.inputs( False, data_dir = Datadir, batch_size = BatchSize )
# images_eval, labels_eval = input10.inputs( True, data_dir = Datadir, batch_size = BatchSize )

x_input = tf.placeholder( tf.float32, shape=[None, 32, 32, 3] )
y_label = tf.placeholder( tf.float32, shape=[None,10] )
trainflag = tf.placeholder(tf.bool)


def batch_norm_layer(value,train = None, name = 'batch_norm'):
  if train is not None:
      return batch_norm(value, decay = 0.9,updates_collections=None, is_training = True)
  else:
      return batch_norm(value, decay = 0.9,updates_collections=None, is_training = False)

#layer1
x_image = tf.reshape(x_input, [-1,32,32,3])

W1 = tf.Variable( initial_value = tf.truncated_normal([5,1,3,64], stddev = 0.001 ), dtype= tf.float32, name = 'W1' )
B1 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32 , name = 'B1')
W1r = tf.Variable( initial_value = tf.truncated_normal([1,5,64,64], stddev = 0.001 ), dtype= tf.float32, name = 'W1' )
B1r = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32 , name = 'B1')

Net1r = tf.nn.relu( batch_norm_layer((tf.nn.conv2d( x_image, W1, strides=[1,1,1,1], padding='SAME' ) + B1), trainflag ))
Pool1r = tf.nn.max_pool( Net1r, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )

Net1 = tf.nn.relu( batch_norm_layer((tf.nn.conv2d( Pool1r, W1r, strides=[1,1,1,1], padding='SAME' ) + B1r),trainflag)) #batch normalization ????????
Pool1 = tf.nn.max_pool( Net1, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )


W2_3 = tf.Variable( initial_value = tf.truncated_normal([3,3,64,64], stddev = 0.001 ), dtype= tf.float32, name = 'W2' )
B2_3 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32, name = 'B2' )
Net2_3 = tf.nn.relu(batch_norm_layer(( tf.nn.conv2d( Pool1, W2_3, strides=[1,1,1,1], padding='SAME' ) + B2_3 ), trainflag))

W2_5 = tf.Variable( initial_value = tf.truncated_normal([5,5,64,64], stddev = 0.001 ), dtype= tf.float32, name = 'W2' )
B2_5 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32, name = 'B2' )
Net2_5 = tf.nn.relu(batch_norm_layer(( tf.nn.conv2d( Pool1, W2_5, strides=[1,1,1,1], padding='SAME' ) + B2_5 ), trainflag))

Net2 = tf.concat([Net2_3, Net2_5],3)

Pool2 = tf.nn.max_pool( Net2, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )

#layer3
W3 = tf.Variable( initial_value = tf.truncated_normal([3,3,128,64], stddev = 0.001 ), dtype= tf.float32, name = 'W3' )
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

lb=0.001
# loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits( labels = y_label, logits = logits, name = 'loss' ))+lb*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W1r)+tf.nn.l2_loss(W2_1)+tf.nn.l2_loss(W2_3)+tf.nn.l2_loss(W2_5)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(WL))
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits( labels = y_label, logits = logits, name = 'loss' ))+lb*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W1r)+tf.nn.l2_loss(W2_3)+tf.nn.l2_loss(W2_5)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(WL))

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

timeSta = time.time()

c10input = C10Input(Datadir)

for i in range(TrainSteps):
    # trainx, trainy = sess.run( [ images,labels ] )
    trainx, trainy = c10input.get_batch_data( BatchSize )
    trainy_b = np.eye(10)[trainy]
    sess.run([step,train], feed_dict={ x_input:trainx, y_label:trainy_b, trainflag:1 })

    if(i%200 == 0):
        evalx, evaly =  c10input.get_batch_data( BatchSize )
        evaly_b = np.eye(10)[evaly]
        print("step:{0}, accuracy 0:{1}".format( i, sess.run( evaluation, feed_dict={x_input:evalx, y_label:evaly_b } ) ))
        print("step:{0}, loss:{1}".format( i, sess.run( loss, feed_dict={x_input:trainx, y_label:trainy_b } ) ))

    # if(i%10 == 0):
    #     result = sess.run(merged, feed_dict = { x_input:trainx, y_label:trainy_b })
    #     writer.add_summary(result, i)
saver.save(sess, "my_net/train_result.ckpt")
timeEnd = time.time()
print("save train result, train time: {} minutes, {} seconds".format( int((timeEnd-timeSta)/60), int((timeEnd-timeSta)%60) ))
