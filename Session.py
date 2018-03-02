
import numpy as np
class Session():
    def __init__(self):
        self.nodes = []

    def add_node(self,node):
        self.nodes.append(node)

    def step(self,x,y):
        for n in self.nodes:
            x = n.forward(x)
           # print x,n.name
        lr = 1#-0.05
        self.nodes.reverse()
        # a fancy way of setting a learning rate
        d=-0.05
        for n in self.nodes:
            print 'backward'
            d_ = n.backward(d)
           # print d_,n.name
            if hasattr(n,'trainable'):
                print 'comp_grad',d.shape
                n.comp_grad(d)
                n.update([lr*g for g in n.grad])
                print 'grad',np.max(n.grad[0])
                #hp.plot(n.grad[0].flatten())
            d= d_
        self.nodes.reverse()
        d = x#-y.reshape(x.shape)
        return d
