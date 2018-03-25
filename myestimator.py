#This file is customer estimator practice
#Create my own estimator
#pre-made Estimators are subclasses of the tf.estimator.Estimator, which completed by others
#Create a customer estimator needs following procedure:
#1, Create an input function ,which is no difference with pre-made estimator use
#2, Create feature columns
#3, Write a model function,this function has different work mode, include
#   training(training the model get hyper parameters), evaluate(evaluate the perfermance of the
#   model), predict(predict the unkown sample)

#Input Function

#feature columns


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
    net = tf.feature_columns.input_layer(features = features, params['feature_columns'] )

    #hidden layers
    for units in params[hidden_layers]:
        net = tf.layers.dense( net, units=units, activation = tf.nn.sigmoid )
    #output layers
    logits = tf.layers.dense( net, params['n_classed'], activation = None ) #no activation function is different with hidden layers
    precated_class = tf.argmax(logits, 1)
    #now handler the result
    if mode == tf.estimator.ModeKeys.PREDICT:   #predict
        predictions = {
          'class_ids':precated_class[:,tf.newaxis],
          'probabilities':tf.nn.softmax(logits),
          'logits':logits,
        }
        return tf.estimator.EstimatorSpec( mode, predictions = predictions )

    #evaluate, return requires loss
    if mode == tf.estimator.ModeKeys.EVAL:

    #training, returning requires loss and train_op




classifier = tf.estimator.Estimator(
    model_fn = my_model_fun,
    #this params functions will pass to model function
    params={
        'feature_columns':feature_columns,
        'hidden_layers':[10,20]
        'n_classed': 3,
    },

)