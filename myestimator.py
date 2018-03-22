#This file is customer estimator practice
#Create my own estimator
#pre-made Estimators are subclasses of the tf.estimator.Estimator, which completed by others
#Create a customer estimator needs following procedure:
#1, Create an input function ,which is no difference with pre-made estimator use
#2, Create feature columns
#3, Write a model function,this function has different work mode, include
#   training(training the model get hyper parameters), evaluate(evaluate the perfermance of the
#   model), predict(predict the unkown sample)