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
yolov1使用的dounding box 一个框只能预测一个物体
'''
import config as cfg
import tensorflow as tf
import tensorflow.contrib.slim as slim

class yl_v1_tiny(object):
  """docstring for ylnet"""
  def __init__(self, arg):
    super(yl_v3_tiny, self).__init__()
    self.arg            = arg
    self.image_size     = cfg.IMAGE_SIZE
    self.cell           = cfg.CELL_SIZE
    self.cell_boxes     = cfg.BOXES_PER_CELL
    self.object_classes = cfg.OBJECT_CLASSES
    self.boundary1      = self.cell*self.cell*self.object_classes   #classes
    self.boundary2      = self.cell*self.cell*self.cell_boxes    #confidence,left are center axis and scale
    self.lambda_coord   = cfg.LAMBDA_COORD
    self.lambda_noobj   = cfg.LAMBDA_NOOBJ
    self.nms_threholds  = cfg.NMS_THREHOLDS
    self.batchsize      = cfg.TRAIN_BATCHSIZE
    self.images         = tf.placeholder( dtype = tf.float32, shape = [None, self.image_size, self.image_size, 3], name='x_input' )
    self.labels         = tf.placeholder( dtype = , shape = [ None, self.cell, self.cell ], name='labels' )
    self.logits         = self.net( )
    self.loss           = self.loss( )

    def get_logits( self ):
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

#相交的时候长宽都是四条线，选中间两条
  def calc_iou( self, label_boxes, logits_boxes ):
    label_x = label_boxes[...,0]
    label_y = label_boxes[...,1]
    label_w = label_boxes[...,2]
    label_h = label_boxes[...,3]

    logits_x = logits_boxes[...,0]
    logits_y = logits_boxes[...,1]
    logits_w = logits_boxes[...,2]
    logits_h = logits_boxes[...,3]

    #没有交集
    if( ( ( label_x-label_w/2 ) - (logits_x+logits_w/2) ) * ( ( label_y - label_h/2 ) - ( logits_y+logits_h/2 ) ) ) return 0
    axis_x = sorted( label_x - label_w/2, label_x + label_w/2, logits_x - logits_w/2, logits_y + logits_w/2 )[1:3]
    axis_y = sorted( label_y - label_h/2, label_y + label_h/2, logits_y - logits_h/2, logits_y + logits_w/2 )[1:3]

    return ((axis_x[1] - axis_x[0])*( axis_y[1] - axis_y[0] )) / ( ( label_w * label_h ) + ( logits_w * logits_h ) )    #7*7*2

#calculate loss
  def loss( self ):
#logits
    logits_class = tf.reshape( self.logits[ :,:self.boundary1 ], [self.batchsize, self.cell, self.cell, self.object_classes] )    #batchsize*7*7*20

    logits_p = tf.reshape( self.logits[ :, self.boundary1:self.boundary2 ], [self.batchsize, self.cell, self.cell, self.cell_boxes] )    #batchsize*7*7*2, object exist or not

    logits_boxes = tf.reshape( self.logits[ :,self.boundary2: ], [self.batchsize, self.cell, self.cell, self.cell_boxes, 4] )    #batchsize*7*7*2*4

#labels
    labels_response = tf.reshape( self.labels[...,0], [self.batchsize, self.cell, self.cell, 1] )   #exist or not exist

    label_boxes = tf.reshape( self.labels[..., 1:4], [self.batchsize, self.cell, self.cell, 1, 4] )
    label_boxes = tf.tile( labels, [1,1,1,2,1] )  #every cell has two boxes

    label_classes = tf.reshape( self.labels[..., 5:], [self.batchsize, self.cell, self.cell, self.object_classes] )

#boxes loss
    box_loss_square = tf.square( tf.sqrt( label_boxes[...,2:3] ) - tf.sqrt( logits_boxes[...,2:3] ) )
    box_loss_delta  = tf.reduce_sum( tf.multiply( box_loss_square, labels_response ) )
    box_loss = self.lambda_coord * box_loss_delta

#confidence loss
    iou = self.calc_iou( label_classes, logits_boxes )     # 7*7*2
    conf_obj_loss = tf.multiply( iou, logits_p )
    conf_obj_loss_tmp = tf.square( tf.substract( labels_response - conf_obj_loss ) )
    conf_obj_loss = tf.multiply( labels_response, conf_obj_loss_tmp )
    conf_obj_loss = tf.reduce_sum( conf_obj_loss )

    mask = tf.subtract( 1 - tf.ones_like( labels_response ) ) # no object is 1, or is 0
    conf_noobj_loss = tf.multiply( mask, conf_obj_loss_tmp )
    conf_noobj_loss = tf.reduce_sum( conf_noobj_loss )
    conf_noobj_loss = self.lambda_noobj*conf_noobj_loss

    conf_loss = conf_obj_loss + conf_noobj_loss

#class loss
    class_loss = tf.sbutract( label_classes - logits_class )
    class_loss = tf.reduce_sum( tf.square( class_loss ), 4 )
    class_loss = tf.multiply( labels_response, class_loss )
    class_loss = tf.reduce_sum( class_loss )

    self.loss = box_loss + conf_loss + class_loss
    return self.loss

  def analysis_logits( self, logits ):
    #check the logits is resonable

    #analysis
    ##class
    class_probability = tf.reshape( logits[,:self.boundary1],[self.batchsize, self.cell, self.cell, self.object_classes] )
    ##confidence
    conf = tf.reshape( logits[,self.boundary1:self.boundary2], [self.batchsize, self.cell, self.cell, self.cell_boxes] )
    ##boxes
    boxes = tf.reshape( logits[,self.boundary2:], [self.batchsize, self.cell, self.cell, self.cell_boxes, 4] )

    return class_probability, conf, boxes

  def get_class_probability( self, cls, conf ):
    cls_shape  = tf.shape( cls ).shape.as_list(  )
    conf_shape = tf.shape( conf ).shape.as_list(  )
    if(len( cls_shape ) <=1) or ( len( conf_shape ) <=1 ):    #cls 和 conf只有一维的话是不对的
      return

    if( cls_shape[:-1] != conf_shape[:-1]) or ( cls_shape[:-2] != self.cell ) or( conf_shape[:-2] != self.cell):
      return

#用多重循环和append拼接起来
    cls_specific1 = np.multiply( cls[-1],conf[-1][0] )
    cls_specific2 = np.multiply( cls[-1],conf[-1][1] )
    cls_specific = np.append( cls_specific1, cls_specific2, -2 )
    return cls_specific      #shape is 32*7*7*2*20

#这个函数的返回结果是[[ 类,[概率,box], [概率,box] ], [ 类,[概率,box], [概率,box] ], [ 类,[概率,box], [概率,box] ]...]这样的
  def do_nms( self, classes_index, classes_probs, filtered_boxes ):
    assert len( classes_index[-2] ) == len( classes_probs[-2] ) and len( classes_index[-2] ) == len( filtered_boxes[-2] )
    result = []     #用来保存结果
    for i in range( self.object_classes ):    #每一种类型遍历一遍
      cls_index = np.argwhere( classed_index )
      if( !len( cls_index ) ):
        continue        #这个类别没有物体存在，直接开始下一类
      prob = classes_probs[cls_index[0], cls_index[1]]  #把这种类型物体的概率值全部找出来, batchsize*1
      boxes = filtered_boxes[cls_index[0], cls_index[1]]  #把这种类型的物体的框框全部找出来，这个是跟上面的概率值一一对应的，batchsize*1
      prob_and_boxes = np.append( prob, boxes, 1 )  #把概率和框框联系到一起，类别是i，合并成[概率，bounding box]，这样子可以排序，并且对应的框框位置信息不回乱

      # 开始真正的nms
      # 排序
      prob_and_boxes = np.sort( prob_and_boxes, 0 )    #[ [概率，box], [概率，box] ]

      # 计算IOU，删除一些框
      single_class_result = [i]      #一个类别的物体全部结果都保存在这个里面
      while prob_and_boxes:
        max_value = prob_and_boxes.pop( 0 )  #把最大的先取出来
        for i in range( len( prob_and_boxes ) ):  #拿剩下的所有的跟取出来的比较
          nms_iou = self.get_nms_iou( max_value, prob_and_boxes[i] )
          if nms_iou > self.nms_threholds:
            prob_and_boxes.remove( prob_and_boxes[i] )  #如果IOU超出阈值，就把这个预测结果从prob_and_boxes里面删掉
        single_class_result.append( max_value )   #for 循环之后，删除掉所有和最大概率预测结果高度重复的预测结果，然后把最大概率预测结果保留

      # while 循环之后把当前类添加到结果中append这样一个元素结果list [ 类，[概率，box], [概率，box] ]
      result.append( single_class_result )

    #找出所有的结果之后返回
    return result


  #得到每个bounding box的class-specific confidence score, 有两个confidence
  #threshold, confidence大于这个threshold的时候才保留
  def predict( self, threshold ):
    logits = self.get_logits( )
    clas, conf, boxes = self.analysis_logits( logits ) #32*7*7*20,  32*7*7*2, 32*7*7*2*4
    #只保留大于threshold的值
    clas_prob = self.get_class_probability( clas, conf )   #shape is 32*7*7*2*20, data type is numpy narray
    if not isinstance( clas_prob, np.ndarray ):
      clas_prob = np.array( clas_prob )   #转换成ndarray类型，不然下面的操作是不可以的

    mask = clas_prob >= threshold         #bool type

    clas_prob = np.multiply( clas_prob, mask )       #大于设定的阈值的预测值才保留下来，其他的全部设置为0 confidence*Pr( clss )大于阀值的就保留下来，小于阀值的就不要了

    #1)获取大于0的值的位置索引
    filters = np.nonzero( clas_prob )    #返回的是一个n*5维度的narray，n是多少个非零的值，5是clas_prob的维数:batchisize, cell, cell, cell_boxes, class,一次只能检测一张，不然这个地方就把整个batchsize混到一起了
    #2)根据1)获取的索引，把所有符合要求的clas_prob提出来，组成一个新的list, 每一个的元素是
    filtered_clas_prob = clas_prob[filters[:,0],filters[:,1],filters[:,2],filters[:,3]]      #batchsize*numbers *20  numbers指的是各种类型一共有多少个物体
    #3)根据1)获取的索引，把所有的框框提取出来，组成一个新的list，2）3）之所以可以组成一个新的list而抛弃原来的7*7信息，是因为有了clas prob和位置就可以预测物体在哪里了。
    filtered_boxes = boxes[filters[:,0],filters[:,1],filters[:,2],filters[:,3]]  #操作的单个元素是box，所以只取到filters[:,3]   #batchsize*numbers*4
        #这里有个问题，会不会一个grid里面有两个物体呢？？会的，可能有两个相同物体，但是不会有两个不同物体，我们是根据confidence*Pr( clss )来判断的，里面一共只有两个confidence，乘以的是相同的Pr( clss )
    assert len( filtered_boxes ) == len( filtered_clas_prob )

    #判断是属于哪些类
    classes_index = filtered_clas_prob.argmax( -1 )       #返回整数，表示第几类,batchsize*numbers*1
    classes_probs = np.amax( -1 )        #把概率值取出来，做nms的时候要用,batchsize*numbers*1
    assert len( classes_index ) == len( filtered_boxes ) and len( classes_boxes ) == len( classes_probs ) #assert 可以理解为，保证后面这个条件是满足的
    #到这个位置，预测出了类别和bounding box，这里的输出没有用softmax

    #do nms，把多余的框框删掉，非极大值抑制
    result_cls, result_cls_prob, result_boxes = self.do_nms( classes_index, classes_probs, filtered_boxes )   #
