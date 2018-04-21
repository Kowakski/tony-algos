'''
@Create a customer estimator for cnn network
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# tfe.enable_eager_execution()
# from tensorflow.python import debug as tf_debug
tf.logging.set_verbosity(tf.logging.INFO)
np.set_printoptions(threshold=np.nan)
# print(ndarray)
#define input function

#define feature columns


#define estimator function
def mod_fn( self, features, labels, mode, params ):
    inputs = tf.reshape(features, [-1, 28, 28, 1])
    tf.logging.log( tf.logging.INFO, inputs )

    #convolution layer
    conv1 = tf.layers.conv2d( inputs = features, filters = 32, kernel_size = [3,3], strides = (1,1), padding = 'valid')
    tf.logging.log( tf.logging.INFO, conv1 )
    #pooling layer
    pool1 = tf.layers.average_pooling2d( inputs = conv1, pool_size = ( 2, 2 ), strides = [1,1], padding='valid', data_format='channels_last', name=None )
    tf.logging.log( tf.logging.INFO, pool1 )


    #convolution layer
    conv2 = tf.layers.conv2d( inputs = pool1, filters = 64, kernel_size = [2,2], strides = [1,1], padding = 'valid' )
    #pooling layer
    pool2 = tf.layers.average_pooling2d( inputs = conv2, pool_size = ( 2, 2 ), strides = [1,1], padding = 'valid' )

    #full connected layer
    FullConnet = tf.reshape( pool2, 7*7*64 )  #7width*7height*64channel

    dense = tf.layers.dense( inputs = FullConnet, units = 7*7*64, activation = tf.nn.sigmoid )

    #softmax layer
    logits = tf.layers.dense( inputs = dense, units = 10 )

    #losses
    loss = tf.tf.losses.sparse_softmax_cross_entropy(labels =  labels, logits = logits, weights = 1.0)

    if mode == tf.estimator.ModeKeys.PREDICT:

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer( learning_rate = 0.3 )
        train_op = optimizer.minize(loss)
        return tf.estimator.EstimatorSpec( mode, loss, train_op = train_op )

    if mode == tf.estimator.ModeKeys.EVAL:

def data_input_fn( self, data, labels )
    return tf.estimator.inputs.numpy_input_fn( x = data, labels = labels )

#define main function
def main(unused_args):
    data = tf.contrib.learn.datasets.mnist.load_mnist()
    TrainImages = data.train.images
    TrainLabels = data.train.labels
    EvaluImages = data.validation.images
    EvaluLabels = data.validation.labels

    tf.logging.log( tf.logging.INFO, TrainImages.shape )
    tf.logging.log( tf.logging.INFO, TrainLabels.shape )

    estimator = tf.estimator.Estimator(
        model_fn = ,
        model_dir = './cnn',
        params = 
        )

    train = estimator.train()

    evalutaion = estimator.evaluate()



if __name__ == "__main__":
    tf.app.run()