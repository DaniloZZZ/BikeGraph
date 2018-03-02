from LayerBase  import NodeBase 
import numpy as np

class Dense(NodeBase):
    def __init__(self, inp,n):
        #inp shape: l,f
        f = inp.shape[-1] # feature count
        self.name = 'dense'
        self.W = np.random.randn(f,n)
        self.b = np.random.randn(1,n)*0
        self.shape = (n,) # neurons count
        self.trainable = (self.W,self.b)
        self.op = lambda x:\
                np.dot(x,self.W)+self.b

    def forward(self,inp):
        self.inp = inp
        return NodeBase.forward(self,self.inp)

    def dop(self,x):
        # input shape - l,n
        # output shape - l,f
        return np.dot(x,self.W.T)

    def comp_grad(self,x):
        # x shape - l,n
        # output shape - f,n
        # this will result in sum of grads for every obj of l
        self.grad = (np.dot(self.inp.T,x)/x.shape[0],np.mean(x))
        print self.grad[0].shape

    def update(self,dv):
        self.W+=dv[0]
        self.b+=dv[1]


