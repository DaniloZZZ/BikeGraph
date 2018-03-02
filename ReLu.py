from LayerBase  import NodeBase 
import numpy as np
class ReLu(NodeBase):
    def __init__(self):
        self.shape=(1,) # this is not quite true
                        # shape is actually same as inp
        self.op =lambda x: np.maximum(x,0)
        self.dop = lambda x: x>0
        self.name = 'relu'

class Square(NodeBase):
    def __init__(self):
        self.op =lambda x: np.square(x)
        self.dop = lambda x: 2*x
        self.name = 'relu'
