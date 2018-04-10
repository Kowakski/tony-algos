'''
@Create a customer estimator for cnn network
'''
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# tfe.enable_eager_execution()
# from tensorflow.python import debug as tf_debug
tf.logging.set_verbosity(tf.logging.INFO)

#define input function

#define feature columns

#define estimator function

#define main function
def main(unused_args):
    data = tf.contrib.learn.datasets.mnist.load_mnist()
    # iterator = data.make_one_shot_iterator()
    # item = iterator.get_next()

    tf.logging.log(tf.logging.INFO, data)
    tf.logging.log(tf.logging.INFO, data.train)
    tf.logging.log(tf.logging.INFO, data.validation)


if __name__ == "__main__":
    tf.app.run()