import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Create Data use numpy
xTrain = np.random.random(50)
noise = np.random.normal(0, 0.1, xTrain.shape).astype(np.float32)
yTrain = 6*xTrain+ 1 + noise  #Get 6 and 1
yWithoutNoise = 6*xTrain+ 1
fig = plt.figure( )
ax1 = fig.add_subplot(111)
ax1.scatter( xTrain, yTrain )
ax1.plot( xTrain, yWithoutNoise, 'r+' )

xInput = tf.placeholder( dtype = tf.float32 )
# xInput = tf.Print(xInput,[xInput], 'xInput:', first_n = 5)

yInput = tf.placeholder( dtype = tf.float32 )

# weight = tf.Variable( 0.3, name='value', dtype = tf.float32 )
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# weight = tf.Print(weight,[weight], 'weights:', first_n = 500)

# bias = tf.Variable( 1, name='bias', dtype = tf.float32 )
bias = tf.Variable(tf.zeros([1]))
bias = tf.Print(bias,[bias], 'bias:', first_n = 5)

OutPut = tf.multiply( weight, xInput ) + bias
# OutPut = tf.Print(OutPut,[OutPut], 'OutPut:', first_n = 5)

loss = tf.reduce_mean( tf.square( OutPut - yInput ) )
loss = tf.Print(loss, [loss], "loss is", summarize = 1000)

optimizer = tf.train.GradientDescentOptimizer( 0.3 )
train = optimizer.minimize(loss)

init = tf.global_variables_initializer( )
sess = tf.Session( )
sess.run(init)

for i in range(200):
    sess.run( train, feed_dict={ xInput:xTrain, yInput:yTrain } )

print("weights is:{0} bias is:{1}".format( sess.run(weight), sess.run(bias) ))
w = sess.run(weight)
b = sess.run(bias)
y = xTrain*w + b
ax1.plot(xTrain, y, 'g')
plt.show( )