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
import img_data
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def main(  ):
  Data = img_data.img_data( 'train' )
  images, labels = Data.get( )
  print( "images {} labels {}".format( np.shape(images), np.shape(labels) ) )

if __name__ == '__main__':
  main(  )