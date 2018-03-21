import csv
import numpy as np
import tensorflow as tf

IRIS_TEST = "iris_test.csv"
IRIS_TRAINING = 'iris_training.csv'

COLUMNS = ['SepalLength','SepalWith','PetalLength','PetalWith','label']
FIELD_DEFAULTS = [[0.0],[0.0],[0.0],[0.0],[0]]

def __parse_line(line):   #"line" is the whole file
  fields = tf.decode_csv( line, FIELD_DEFAULTS )
  features = dict( zip( COLUMNS, fields ) )
  label = features.pop('label')
  return features, label

def data_input_function( PATH, BatchSize ):
    Data = tf.data.TextLineDataset( PATH, compression_type = None, buffer_size = None ).skip(1)
    Data = Data.map( __parse_line )
    Data = Data.shuffle( 1000 ).repeat( ).batch( BatchSize )
    return Data
#Define input function
# debuf:<class 'tensorflow.python.data.ops.readers.TextLineDataset'>

keys = ['SepalLength', 'SepalWith', 'PetalLength', 'PetalWith']

FeaturesColumns = [ tf.feature_column.numeric_column( key = s, shape=(1,), default_value=None, dtype = tf.float32, normalizer_fn = None ) for s in keys ]
print("\r\n Get features:{0}".format( FeaturesColumns ) )

classifier = tf.estimator.DNNClassifier( hidden_units = [10, 20], feature_columns = FeaturesColumns, model_dir = "./storage", n_classes=3, weight_column = None )
print("\r Create classifier successfully\n")

classifier.train( input_fn = lambda: data_input_function( IRIS_TRAINING, 20 ), hooks = None, steps = 200, max_steps = None, saving_listeners =  None )
print("\r Train successfully\n")

EvaResult = classifier.evaluate( input_fn = lambda: data_input_function( IRIS_TEST, 20 ), steps=1, hooks=None, checkpoint_path=None, name=None )

print("\r\n EvaResult is:{0}\r\n".format(EvaResult))
# init =  tf.global_variables_initializer()
# sess = tf.Session()

# sess.run(init)