'''
layer     filters    size              input                output
    0 conv     16  3 x 3 / 1   448 x 448 x   3   ->   448 x 448 x  16  0.173 BFLOPs
    1 max          2 x 2 / 2   448 x 448 x  16   ->   224 x 224 x  16
    2 conv     32  3 x 3 / 1   224 x 224 x  16   ->   224 x 224 x  32  0.462 BFLOPs
    3 max          2 x 2 / 2   224 x 224 x  32   ->   112 x 112 x  32
    4 conv     64  3 x 3 / 1   112 x 112 x  32   ->   112 x 112 x  64  0.462 BFLOPs
    5 max          2 x 2 / 2   112 x 112 x  64   ->    56 x  56 x  64
    6 conv    128  3 x 3 / 1    56 x  56 x  64   ->    56 x  56 x 128  0.462 BFLOPs
    7 max          2 x 2 / 2    56 x  56 x 128   ->    28 x  28 x 128
    8 conv    256  3 x 3 / 1    28 x  28 x 128   ->    28 x  28 x 256  0.462 BFLOPs
    9 max          2 x 2 / 2    28 x  28 x 256   ->    14 x  14 x 256
   10 conv    512  3 x 3 / 1    14 x  14 x 256   ->    14 x  14 x 512  0.462 BFLOPs
   11 max          2 x 2 / 2    14 x  14 x 512   ->     7 x   7 x 512
   12 conv   1024  3 x 3 / 1     7 x   7 x 512   ->     7 x   7 x1024  0.462 BFLOPs
   13 conv    256  3 x 3 / 1     7 x   7 x1024   ->     7 x   7 x 256  0.231 BFLOPs
   14 connected                            12544  ->  1470
   15 Detection Layer
forced: Using default '0'
Loading weights from darknet53.conv.74...Done!
'''
import config as cfg
import tensorflow as tf
import tensorflow.contrib.slim as slim

class yl_v1_tiny(object):
  """docstring for ylnet"""
  def __init__(self, arg):
    super(yl_v3_tiny, self).__init__()
    self.arg        = arg
    self.image_size = cfg.IMAGE_SIZE
    self.cell       = cfg.CELL_SIZE
    self.cell_boxes = cfg.BOXES_PER_CELL
    self.object_classes = cfg.OBJECT_CLASSES
    self.boundary1  = self.cell*self.cell*self.object_classes   #classes
    self.boundary2  = self.cell*self.cell*self.cell_boxes    #confidence,left are center axis and scale
    self.lambda_coord = cfg.LAMBDA_COORD
    self.lambda_noobj = cfg.LAMBDA_NOOBJ
    self.batchsize = cfg.TRAIN_BATCHSIZE
    self.images = tf.placeholder( dtype = tf.float32, shape = [None, self.image_size, self.image_size, 3], name='x_input' )
    self.labels = tf.placeholder( dtype = , shape = [ None, self.cell, self.cell ], name='labels' )
    self.logits = self.net( )
    self.loss = self.loss( )

    def net( self ):
'''
convolution(inputs,
          num_outputs,
          kernel_size,
          stride=1,
          padding='SAME',
          data_format=None,
          rate=1,
          activation_fn=nn.relu,
          normalizer_fn=None,
          normalizer_params=None,
          weights_initializer=initializers.xavier_initializer(),
          weights_regularizer=None,
          biases_initializer=init_ops.zeros_initializer(),
          biases_regularizer=None,
          reuse=None,
          variables_collections=None,
          outputs_collections=None,
          trainable=True,
          scope=None):
'''

        net1  = slim.conv2d( self.images, num_outputs=16, kernel_size = [3,3], stride = [1,1], padding='SAME' )
        pool1 = slim.max_pool2d( net1, [2,2], stride = 2 )

        net2  = slim.conv2d( pool1, num_outputs = 32, kernel_size = [3,3], stride = [1,1], padding='SAME' )
        pool2 = slim.max_pool2d( net2, [2,2], stride = 2 )

        net3  = slim.conv2d( pool2, num_outputs = 64, kernel_size= [3,3], stride = [1,1], padding = 'SAME' )
        pool3 = slim.max_pool2d( net3, [2,2], stride = 2 )

        net4 = slim.conv2d( pool3, num_outputs = 128, kernel_size = [3,3], stride = [1,1], padding = 'SAME' )
        pool4 = slim.max_pool2d( net4, [2,2], stride = 2 )

        net5 = slim.conv2d( pool4, num_outputs = 256, kernel_size = [3,3], stride = [1,1], padding = 'SAME' )
        pool5 = slim.max_pool2d( net5, [2,2], stride = 2 )

        net6 = slim.conv2d( pool5, num_outputs = 512, kernel_size = [3,3], stride = [1,1], padding = 'SAME' )
        pool5 = slim.max_pool2d( net6, [2,2], stride = 2 )

        net7 = slim.conv2d( pool5, num_outputs = 1024, kernel_size = [3,3], stride = [1,1], padding = 'SAME' )
        net8 = slim.conv2d( net7, num_outputs = 256, kernel_size = [3,3], stride = [1,1], padding = 'SAME' )

'''
x_image = tf.reshape(x,[-1,28,28,1])
    hidden_1 = slim.conv2d(x_image,5,[5,5])
shape_h1 = tf.shape( hidden_1 )
    pool_1 = slim.max_pool2d(hidden_1,[2,2])
    hidden_2 = slim.conv2d(pool_1,5,[5,5])
    pool_2 = slim.max_pool2d(hidden_2,[2,2])
    hidden_3 = slim.conv2d(pool_2,20,[5,5])
    hidden_3 = slim.dropout(hidden_3,keep_prob)
    out_y = slim.fully_connected(slim.flatten(hidden_3),10,activation_fn=tf.nn.softmax)
'''

#reshape net8 & full connected layer
        logits = slim.fully_connected( slim.flatten( net8 ), 1470, activation_fn=tf.nn.softmax )
        return logits


#calculate loss
  def loss( self ):
#logits
    logits_class = tf.reshape( self.logits[ :,:self.boundary1 ], [self.batchsize, self.cell, self.cell, self.object_classes] )    #batchsize*7*7*20

    logits_confidence = tf.reshape( self.logits[ :, self.boundary1:self.boundary2 ], [self.batchsize, self.cell, self.cell, self.cell_boxes] )    #batchsize*7*7*2

    logits_boxes = tf.reshape( self.logits[ :,self.boundary2: ], [self.batchsize, self.cell, self.cell, self.cell_boxes, 4] )    #batchsize*7*7*2*4

#labels
    labels_response = tf.reshape( self.labels[...,0], [self.batchsize, self.cell, self.cell, 1] )   #exist or not exist

    label_boxes = tf.reshape( self.labels[..., 1:4], [self.batchsize, self.cell, self.cell, 1, 4] )
    label_boxes = tf.tile( labels, [1,1,1,2,1] )  #every cell has two boxes

    label_classes = tf.reshape( self.labels[..., 5:], [self.batchsize, self.cell, self.cell, self.object_classes] )

#boxes loss
    box_loss_square = tf.square( tf.sqrt( label_boxes[...,2:3] ) - tf.sqrt( logits_boxes[...,2:3] ) )
    box_loss_delta  = tf.reduce_sum( tf.maltiply( box_loss_square, labels_response ) )
    box_loss = self.lambda_coord * box_loss_delta

