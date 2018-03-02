import numpy as np
class NodeBase:
    def __init__(self,mat):
        self.shape = mat.shape

    def forward(self,inp):
        res = self.op(inp)
        self.cache = inp
        return res

    def backward(self,reinp):
        if self.shape[0]==1:
            # when jacobian is diagonal
            return self.dop(self.cache)*reinp
        else:
            return np.dot(reinp,self.dop(self.cache))

