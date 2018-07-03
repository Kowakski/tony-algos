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
        if self.FileIndex > 5:
            self.FileIndex = 1
        return X,Y

    """docstring for C10Input"""
    def __init__( self, DataPath ):
        super(C10Input, self).__init__()
        assert DataPath is not None
        self.index     = 0
        self.Path      = DataPath
        self.FileIndex = 1

    def merge_channel( self, Data ):
        assert len(np.shape(Data)) == 4
        r = Data[:,0][:,:,:,np.newaxis]    #batchsize*32*32
        g = Data[:,1][:,:,:,np.newaxis]
        b = Data[:,2][:,:,:,np.newaxis]

        result = np.append( r,g, axis = 3 )
        result = np.append( result, b, axis = 3 )

        return result

    def get_batch_data(self, batchsize):
        if(self.index == 0):
            self.data, self.label = self.get_data(self.Path)

        # print(self.index)
        if((self.index + batchsize) <= 10000):
            # print("here 1")
            batch_data, batch_label = self.data[ self.index: self.index+batchsize ], self.label[ self.index: self.index+batchsize ]
            self.index = self.index+batchsize
            if(self.index == 10000):
                self.index = 0

        if(self.index<10000) and (self.index + batchsize) > 10000:
            # print("here 2")
            batch_data, batch_label = self.data[self.index : 9999], self.label[self.index:9999]
            self.data, self.label = self.get_data( self.Path )
            self.index = self.index + batchsize - 9999  #batchsize - (9999-self.index)
            batch_data = np.append( batch_data, self.data[0:self.index] )
            batch_label = np.append( batch_label, self.label[0:self.index] )

        # batch_data = np.reshape( batch_data, [batchsize, 32, 32, 3] )
        batch_data = batch_data.reshape( [batchsize, 3, 32, 32] )
        batch_data = self.merge_channel(batch_data)
        # batch_label = np.reshape( batch_label, [batchsize, 1] )

        return batch_data, batch_label