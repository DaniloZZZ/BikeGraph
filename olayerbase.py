import numpy as np
class NodeBase:
    def __init__(self,mat):
        self.shape = mat.shape

    def forward(self,inp):
        res = self.op(inp)
        self.cache = res
        return res

    def backward(self,reinp):
        return self.dop(self.cache)*reinp

