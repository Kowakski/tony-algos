import numpy as np

class vgg19(self):
    def __init__( self, vars_path = None )：
        if vars_path is not None:
            self.vars_dict = np.load(vars_path, encoding='latin1').item()
        else:
            self.vars_dict = {}

    def get_var( self, shape = [], lyrname = None, varname = None ):
        initiate_value = self.vars_dict[lyrname][idx]    #get weight from pre-trained model

        if initiate_value is None:   #no pre trained weights, initiate one
            initiate_value = tf.truncated_normal( shape, main = 0.0, stddev = 1.0 )
        else:
            assert (shape == np.shape(initiate_value))

        var = tf.Variable( initiate_value, name = lyrname + varname )

        return var

    def conv( self, bottom = [], shape = [ ], outchannel = 0, lyername = none ):
        filter_var = self.get_var( shape, lyername, "_filter" )
        bais_var = self.get_var( shape, lyername, "_bias" )
        result = tf.nn.bias_add(tf.nn.conv2d( bottom, filter_var, [1,1,1,1], padding = "SAME" ), bais_var)
        result = tf.relu(result)
        return result

    def maxpool( self, bottom):
        result = tf.nn.maxpool( bottom, [1,2,2,1], [1,2,2,1], padding = 'SAME' )
        return result

    def fu_layer_mul( ly = None, shape, name ):
        weights_var = self.get_var( shape, lyername, "_fc_weights" )
        bias_fc = self.get_var( shape,  )
    def build(self, rgb, ):
        # filter size is 3*3 and stride is 1
        # conv3_64*2 + maxpool
        conv1_1  = self.conv( rgb, [3, 3, 3, 64], 'conv1_1' )    #[3,3,3,64] [inchannel, wide, height, outchannel]
        conv1_2  = self.conv( conv1_1, [3,3,3,64], 'conv1_2' )
        maxpool1 = self.maxpool( conv1_2, "pool1" )


        # conv3_128*2 + maxpool
        conv2_1  = self.conv(maxpool1, [64,3,3,128], "conv2_1")
        conv2_2  = self.conv(maxpool1, [64,3,3,128], "conv2_1")
        maxpool2 = self.maxpool(conv2_2, "pool2")

        # conv3_256 * 4 + maxpool*1
        conv3_1  = self.conv(maxpool2, [128,3,3,256], "conv3_1")
        conv3_2  = self.conv(maxpool2, [128,3,3,256], "conv3_2")
        conv3_3  = self.conv(maxpool2, [128,3,3,256], "conv3_3")
        conv3_4  = self.conv(maxpool2, [128,3,3,256], "conv3_4")
        maxpool3 = self.maxpool( conv3_4, "pool3" )

        # conv3_512 * 4 + maxpool*1
        conv4_1  = self.conv(maxpool2, [ 256,3,3,512 ], "conv4_1")
        conv4_2  = self.conv(maxpool2, [ 256,3,3,512 ], "conv4_2")
        conv4_3  = self.conv(maxpool2, [ 256,3,3,512 ], "conv4_3")
        conv4_4  = self.conv(maxpool2, [ 256,3,3,512 ], "conv4_4")
        maxpool4 = self.maxpool( conv4_4, "pool4" )

        # conv3_512 *4 + maxpool*1
        conv5_1  = self.conv(maxpool2, [ 512,3,3,512 ], "conv5_1")
        conv5_2  = self.conv(maxpool2, [ 512,3,3,512 ], "conv5_2")
        conv5_3  = self.conv(maxpool2, [ 512,3,3,512 ], "conv5_3")
        conv5_4  = self.conv(maxpool2, [ 512,3,3,512 ], "conv5_4")
        maxpool5 = self.maxpool( conv5_4, "pool5" )

        # fc4096*2 + fc1000 + soft-max
        fc_in = tf.reshape(maxpool5, [1,])
        fc_1 = self.fu_layer_mul(fc_in, [len(fc_in), 4096], "fc_1")
        fc_2 = self.fu_layer_mul(fc_in, [4096, 4096], "fc_2")