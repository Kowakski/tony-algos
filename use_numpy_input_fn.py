import csv
import numpy as np
import tensorflow as tf

TrainDataFlag = 0
TestDataFlag = 0

IRIS_TEST = "iris_test.csv"

#Define input function

def InputTestData():
    global TestDataFlag
    with open('./iris_test.csv') as csvfile:
      csvf = csv.reader( csvfile )
      x = []
      y = []
      for row in csvf:
        if TestDataFlag is 0:
          TestDataFlag = 1
          continue
    
        templist = []
        for CharValue in row:
          templist.append(float(CharValue))
        x.append(templist[:-1])
        y.append(int(templist[-1]))

#Return result
      return tf.estimator.inputs.numpy_input_fn(
        x = { "xLabel":np.array(x) },
        y = np.array(y),
        num_epochs = 1,
        shuffle = False )

#Define input function
def InputTrainData():
    global TrainDataFlag
    with open('./iris_training.csv') as csvfile:
      csvf = csv.reader( csvfile )
      x = []
      y = []
      for row in csvf:
        if TrainDataFlag is 0:
          TrainDataFlag = 1
          continue
    
        templist = []
        for CharValue in row:
          templist.append(float(CharValue))
        x.append(templist[:-1])
        y.append(int(templist[-1]))
      
      return tf.estimator.inputs.numpy_input_fn(
        x={"xLabel":np.array(x)},
        y=np.array(y),
        num_epochs=None,
        shuffle=True )

#Define feature columns
features_columns = [tf.feature_column.numeric_column('xLabel', shape=[4] )]

#Define estimator
estimator = tf.estimator.DNNClassifier( feature_columns = features_columns, hidden_units=[10, 20, 10], n_classes = 3, optimizer = 'Adagrad' )

#train
estimator.train( input_fn = InputTrainData(), hooks = None, steps = 2000, saving_listeners = None )

#Define evaluate
accuracy_score = estimator.evaluate( input_fn = InputTestData() )["accuracy"]

print("\r\nTest Accuracy:{0}\r\n".format(accuracy_score))

