import tensorflow as tf
import numpy  as np
import pickle as p

class C10Input(object):
    def get_data(self, DataPath):
        f = open(DataPath + '/data_batch_%d' % self.FileIndex, 'rb')
        datadict = p.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        self.FileIndex = self.FileIndex + 1
        return X,Y

    """docstring for C10Input"""
    def __init__( self, DataPath ):
        super(C10Input, self).__init__()
        assert DataPath is not None
        self.index     = 0
        self.Path      = DataPath
        self.FileIndex = 1

    def get_batch_data(self, batchsize):
        if(self.index == 0):
            data, label = self.get_data(self.Path)

        if((self.index + batchsize) < 10000):
            batch_data, batch_label = data[ self.index: self.index+batchsize ], label[ self.index: self.index+batchsize ]
            self.index = self.index+1

        if(self.index<10000) and (self.index + batchsize) > 10000:
            batch_data, batch_label = data[self.index : 9999], label[self.index:9999]
            data, label = self.get_data( )
            self.index = self.index + batchsize - 9999  #batchsize - (9999-self.index)
            batch_data = np.append( batch_data, data[0:self.index] )
            batch_label = np.append( batch_label, label[0:self.index] )

        # batch_data = np.reshape( batch_data, [batchsize, 32, 32, 3] )
        batch_data = batch_data.reshape( [batchsize, 32, 32, 3] )
        batch_label = np.reshape( batch_label, [batchsize, 1] )

        return batch_data, batch_label