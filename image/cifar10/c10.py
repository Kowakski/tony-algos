#This is classification for 10 animals:
import sys
import numpy as np
# from cifar10 import cifar10_input
import cifar10.cifar10_input as input10
import tensorflow as tf
from functools import reduce

Datadir = 'D:\\tmp\\cifar10_data\\cifar-10-batches-bin'

BatchSize = 128
TrainSteps = 1500

images, labels = input10.inputs( False, 'D:\\tmp\\cifar10_data\\cifar-10-batches-bin', BatchSize )

x_input = tf.placeholder( tf.float32, shape=[None, 24, 24, 3] )
y_label = tf.placeholder( tf.float32, shape=[None,10] )

W1 = tf.Variable( initial_value = tf.truncated_normal([5,5,3,64]), dtype= tf.float32 )
B1 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32 )
tf.summary.histogram("weights1",W1)
tf.summary.histogram("bias1",B1)

x_image = tf.reshape(x_input, [-1,24,24,3])
print("x_image {}".format(x_image))

Net1 = tf.nn.relu( tf.nn.conv2d( x_image, W1, strides=[1,2,2,1], padding='SAME' ) + B1 )
print("Net1 {}".format(Net1))
# Net1 = tf.Print( Net1, [Net1], "Net1 is:", summarize=1000 )

Pool1 = tf.nn.max_pool( Net1, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )
# Pool1 = tf.Print( Pool1, [Pool1], "Pool1 is:", summarize=1000 )
print("Pool1 {}".format(Pool1))



W2 = tf.Variable( initial_value = tf.truncated_normal([3,3,64,64]), dtype= tf.float32 )
B2 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32 )

Net2 = tf.nn.relu( tf.nn.conv2d( Pool1, W2, strides=[1,2,2,1], padding='SAME' ) + B2 )
print("Net2 {}".format(Net2))
# Net2 = tf.Print( Net2, [Net2], "Net2 is:", summarize=1000 )

Pool2 = tf.nn.max_pool( Net2, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )
print("Pool2 {}".format(Pool2))
# Pool2 = tf.Print( Pool2, [Pool2], "Pool2 is:", summarize=1000 )

# nod = reduce(mul, Pool2)  #calculate how many nods
nod = 1
# for i in range( np.shape(Pool2)[1:] ):
for i in Pool2.shape[1:]:
    nod = nod * int(i)

FullayerInput = tf.reshape(Pool2, [-1,nod])
print("FullayerInput {}".format(FullayerInput))

'''
WF = tf.Variable( initial_value = tf.truncated_normal([nod,20]), dtype = tf.float32 )  #Weights of full connected layer
BF = tf.Variable( initial_value = tf.zeros([1, 20]), dtype = tf.float32 )
FullLayerOut = tf.nn.relu(tf.matmul(FullayerInput, WF) + BF)
print("FullLayerOut is {0} //{1}".format( np.shape(FullLayerOut), FullLayerOut ) )

WL = tf.Variable( initial_value = tf.truncated_normal([20,10]), dtype = tf.float32 )  #weights of logic layer
BL = tf.Variable( initial_value = tf.zeros([1, 10]), dtype = tf.float32 )
logits = tf.nn.relu( tf.matmul(FullLayerOut, WL ) + BL )
print("Logits is:{0}".format(np.shape(logits)) )
'''

WL = tf.Variable( initial_value = tf.truncated_normal([nod,10]), dtype = tf.float32 )  #weights of logic layer
BL = tf.Variable( initial_value = tf.constant(0.1, shape = [1, 10]), dtype = tf.float32 )
logits = tf.nn.relu( tf.matmul(FullayerInput, WL ) + BL )
print("logits {}".format(logits))
# logits = tf.Print( logits, [logits], "logits is:", summarize=1000 )

softmax = tf.nn.softmax( logits )
print("softmax {}".format(softmax))
# softmax = tf.Print( softmax, [softmax], "softmax is:", summarize=1000 )
EvalLabel =  tf.argmax( softmax, 1 )
evaluation = tf.reduce_mean( tf.cast( tf.equal( tf.argmax(y_label,1), EvalLabel ), tf.float32 ) )

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits( labels = y_label, logits = logits, name = 'loss' ))
# loss = -tf.reduce_sum(y_label*tf.log(softmax))
# loss = tf.Print(loss, [loss], message="loss value", summarize = 1000)
print("loss is {0}".format(loss))
tf.summary.scalar('loss',loss)

# optimizer = tf.train.GradientDescentOptimizer(1e-4)
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize( loss )

sess = tf.Session( )
merged = tf.summary.merge_all( )
writer = tf.summary.FileWriter( "./logs",sess.graph )
sess.run( tf.global_variables_initializer( ) )

tf.train.start_queue_runners(sess = sess)

for i in range(TrainSteps):
    trainx, trainy = sess.run( [ images,labels ] )
    trainy_b = np.eye(10)[trainy]
    sess.run(train, feed_dict={ x_input:trainx, y_label:trainy_b })

    if(i%200 == 0):
        evalx, evaly =  sess.run( [ images, labels ] )
        evaly_b = np.eye(10)[evaly]
        print("step:{0}, accuracy:{1}".format( i, sess.run( evaluation, feed_dict={x_input:evalx, y_label:evaly_b} ) ))

    if(i%10 == 0):
        result = sess.run(merged, feed_dict = { x_input:trainx, y_label:trainy_b })
        writer.add_summary(result, i)