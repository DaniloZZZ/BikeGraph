import numpy as np
from LayerBase import NodeBase
class subtr(NodeBase):
    def __init__(self,x,y):
        self.shape = (x.shape[0],)
        self.name = 'subtr'
        self.y = y
        self.op = lambda x:x-self.y
        self.dop= lambda x:1

class add(NodeBase):
    def __init__(self,x,y):
        self.shape = (x.shape[0],)
        self.op = lambda x,y:x+y
        self.dop= lambda x:1
