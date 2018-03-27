#This file is customer estimator practice
#Create my own estimator
#pre-made Estimators are subclasses of the tf.estimator.Estimator, which completed by others
#Create a customer estimator needs following procedure:
#1, Create an input function ,which is no difference with pre-made estimator use
#2, Create feature columns
#3, Write a model function,this function has different work mode, include
#   training(training the model get hyper parameters), evaluate(evaluate the perfermance of the
#   model), predict(predict the unkown sample)

import tensorflow as tf
# from tensorflow.python import debug as tf_debug

#Input Function
COLUMNS = ['SepalLength', 'SepalWith', 'PetalLength', 'PetalWith', 'label']
FIELD_DEFAULTS = [[0.0],[0.0],[0.0],[0.0],[0]]

TrainDataPath = './woodata/iris_training.csv'
EvaluDataPath = './woodata/iris_test.csv'

def __parse_line(line):
    fields = tf.decode_csv( line, FIELD_DEFAULTS )
    features = dict(zip(COLUMNS, fields))
    labels = features.pop('label')
    return features, labels

def data_input_function(PATH, BatchSize):
    Data = tf.data.TextLineDataset( PATH, compression_type = None, buffer_size = None ).skip(1)
    Data = Data.map(__parse_line)
    Data = Data.shuffle(1000).repeat().batch(BatchSize)
    return Data

#feature columns
feature_columns = [tf.feature_column.numeric_column(key=x, shape=(1,),default_value = None, dtype = tf.float32, normalizer_fn=None) for x in COLUMNS[:-1]]

#model function
'''
@comment:
@parameters:
    features, This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode,     # An instance of tf.estimator.ModeKeys, ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT
    params,   # Additional configuration
'''
def my_model_fun( features, labels, mode, params ):
    #input layer, convert feature dictionary to feature
    net = tf.feature_column.input_layer(features = features, feature_columns = params['feature_columns'] )      #!!! important API
    net = tf.Print(net, [net], message ="Input net", first_n = 1, summarize = 10000)
    # print("input feature is:{0}\r\n".format(features) )

    #hidden layers
    for units in params['hidden_layers']:
        net = tf.layers.dense( net, units=units, activation = tf.nn.sigmoid )       #!!! Important API
    #output layers
    logits = tf.layers.dense( net, params['n_classed'], activation = None ) #no activation function is different with hidden layers
    precated_class = tf.argmax(logits, 1)
    # print("precated_class is:{0}\r\n".format(precated_class))

    #now handler the result
    if mode == tf.estimator.ModeKeys.PREDICT:   #predict
        predictions = {
          'class_ids':precated_class[:,tf.newaxis],
          'probabilities':tf.nn.softmax(logits),
          'logits':logits,
        }
        return tf.estimator.EstimatorSpec( mode, predictions = predictions )

    labels = tf.Print(labels, [labels], message = "Cross entropy label is", first_n=1, summarize=10000)
    loss = tf.losses.sparse_softmax_cross_entropy( labels = labels, logits = logits ) #!!! Important API

    #evaluate, return requires loss
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy( labels = labels, predictions = precated_class, name = 'acc_op' )
        metrics = {'accuracy':accuracy}
        return tf.estimator.EstimatorSpec( mode,  loss = loss, eval_metric_ops = metrics )

    #training, returning requires loss and train_op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer( learning_rate = 0.3 )
        train_op = optimizer.minimize( loss, global_step = tf.train.get_global_step() )
        return tf.estimator.EstimatorSpec( mode, loss = loss, train_op = train_op )

# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
classifier = tf.estimator.Estimator(
    model_fn = my_model_fun,
    #this params functions will pass to model function
    params={
        'feature_columns':feature_columns,
        'hidden_layers':[10,20],
        'n_classed': 3,
    },

)
classifier.train(input_fn = lambda: data_input_function(TrainDataPath, 200), hooks=None, steps=200)

evaluation = classifier.evaluate(input_fn = lambda: data_input_function(EvaluDataPath, 200), steps=1, name='evaluation')
print("evaluation is {0}\r\n".format( evaluation ))