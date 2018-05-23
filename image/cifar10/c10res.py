import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm
from PIL import Image as mping

args = sys.argv[1:]
if len(args) == 0:
    print("An image should be as input...")
    exit()
if len(args) == 1:
    image = mping.open(args[0])
    image = image.resize((24,24))
    image = np.array(image)
    # image = tf.image.per_image_standardization(image)
    # image = tf.reshape(image, [1,24,24,3])
    image = image.reshape((1,24,24,3))

x_input = tf.placeholder( tf.float32, shape=[None, 24, 24, 3] )

def batch_norm_layer(value,train = None, name = 'batch_norm'):
    return batch_norm(value, decay = 0.9,updates_collections=None, is_training = False)

W1 = tf.Variable( initial_value = tf.truncated_normal([5,1,3,64], stddev = 0.001 ), dtype= tf.float32, name = 'W1' )
B1 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32 , name = 'B1')
W1r = tf.Variable( initial_value = tf.truncated_normal([1,5,64,64], stddev = 0.001 ), dtype= tf.float32, name = 'W1' )
B1r = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32 , name = 'B1')

x_image = tf.reshape(x_input, [-1,24,24,3])
x_image = tf.map_fn(lambda x: tf.image.per_image_standardization(x), x_image)
# x_image = tf.image.per_image_standardization(x_image)

Net1r = tf.nn.relu( batch_norm_layer((tf.nn.conv2d( x_image, W1, strides=[1,1,1,1], padding='SAME' ) + B1) ))
Pool1r = tf.nn.max_pool( Net1r, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )

Net1 = tf.nn.relu( batch_norm_layer((tf.nn.conv2d( Pool1r, W1r, strides=[1,1,1,1], padding='SAME' ) + B1r))) #batch normalization 是针对激活函数的
Pool1 = tf.nn.max_pool( Net1, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )


#layer2, 多通道卷积
W2_1 = tf.Variable( initial_value = tf.truncated_normal([1,1,64,64], stddev = 0.001 ), dtype= tf.float32, name = 'W2' )
B2_1 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32, name = 'B2' )
Net2_1 = tf.nn.relu(batch_norm_layer(( tf.nn.conv2d( Pool1, W2_1, strides=[1,1,1,1], padding='SAME' ) + B2_1 )))
# Pool2_1 = tf.nn.max_pool( Net2, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME' )

W2_3 = tf.Variable( initial_value = tf.truncated_normal([3,3,64,64], stddev = 0.001 ), dtype= tf.float32, name = 'W2' )
B2_3 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32, name = 'B2' )
Net2_3 = tf.nn.relu(batch_norm_layer(( tf.nn.conv2d( Pool1, W2_3, strides=[1,1,1,1], padding='SAME' ) + B2_3 )))

W2_5 = tf.Variable( initial_value = tf.truncated_normal([5,5,64,64], stddev = 0.001 ), dtype= tf.float32, name = 'W2' )
B2_5 = tf.Variable( initial_value = tf.constant(0.1, shape = [64]), dtype= tf.float32, name = 'B2' )
Net2_5 = tf.nn.relu(batch_norm_layer(( tf.nn.conv2d( Pool1, W2_5, strides=[1,1,1,1], padding='SAME' ) + B2_5 )))

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
print( "softmax is ",softmax )
EvalLabel =  tf.argmax( softmax, 1 )


sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'my_net/train_result.ckpt')

def kind(var):
    return{
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
    }[var]

lable = sess.run( EvalLabel, feed_dict = {x_input:image} )
lable = int(lable)
print("input is ", kind(lable))

# print( "evaluate this is:", sess.run( EvalLabel, feed_dict = {x_input:image} ) )