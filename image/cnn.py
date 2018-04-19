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
    #convolution layer
    net = tf.layers.Conv2D.conv2d(inputs = features, filters = , kernel_size = , stride = (1,1), padding = 'valid'):

    #pooling layer
    net = tf.layers.AveragePooling2D.average_pooling2d( inputs = , pool_size = , strides = , padding='valid', data_format='channels_last', name=None )

    #convolution layer

    #pooling layer

    #full connected layer

    #softmax layer

#define main function
def main(unused_args):
    data = tf.contrib.learn.datasets.mnist.load_mnist()
    TrainImages = data.train.images
    TrainLabels = data.train.labels
    EvaluImages = data.validation.images
    EvaluLabels = data.validation.labels

    print((TrainImages.shape))
    print((TrainLabels.shape))

    tf.logging.log( tf.logging.INFO, TrainImages[:1] )
    tf.logging.log( tf.logging.INFO, TrainLabels[:1] )

if __name__ == "__main__":
    tf.app.run()