import numpy as np
from LayerBase import NodeBase

class MeanSqrError(NodeBase):
    def __init__(self,x,axis=0):
        # input shape arbitraryi, output shape 1
        self.shape = (1,)
        self.name = 'MSE'
        # remember shape for production dout/dinp vector
        self.inpshape = (x.shape[0],)
        self.axis =axis
        self.op = lambda x:np.mean(np.square(x),axis=self.axis)
        self.dop= lambda x:np.ones(self.inpshape)/self.inpshape*2*x
        


