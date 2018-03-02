from ReLu import ReLu,Square
from Dense import Dense
from Session import Session
from basicOps import subtr
from losses import MeanSqrError

import numpy as np
import hello
import time,os
import tensorflow as tf
import hipsterplot as hp
import pandas as pd
import matplotlib.pyplot as plt


# defining a data generation functions
def gen_wine():
    d = pd.read_csv('winequality-red.csv')
    k=d.keys()
    x=np.array(d[k[:-1]])
    y=np.array(d[k[-1]])
    
    mn = np.mean(x,axis=0)
    x-=mn
    x/=np.std(x,axis=0)
    print 'y mean, ystd:',np.mean(y),np.std(y)
    return x,y.reshape(len(y),1)

def gen_reg():
    x = np.random.rand(2000,45)
    y = .3*x[:,0]+x[:,2]+x[:,3]
    return x,np.ones(2000).reshape(2000,1)

def gen_class():
    x = np.random.rand(10,2)+1.
    x2 = np.random.rand(10,2)
    x2[:,0]-=1
    x = np.concatenate((x,x2))

    y = np.concatenate((np.ones(10),np.zeros(10)))
    _y =np.zeros((y.shape[0],2))
    print "\ny",y
    plt.scatter(x[:,0],x[:,1],c=y)
    _y[np.arange(y.shape[0]),y.astype(np.int8)] = 1
    y = _y.T.reshape(10,1)
    return x,y


def run_tf(x,y,epochs=5):
    # Configure Tensorflow session
    fnum =11
    init = tf.contrib.layers.xavier_initializer()
    _x = tf.placeholder(tf.float32, shape=(None,fnum), name='data')
    _y = tf.placeholder(tf.float32, shape=(None,1), name='labels')
    d = tf.layers.Dense(1,
                        kernel_initializer=init,
                        activation=tf.keras.activations.relu)
    ds =d.apply(_x)
    '''
    d2 = tf.layers.Dense(1,
                        kernel_initializer=init,
                        activation=tf.keras.activations.relu)
    ds =d2.apply(ds)
    '''
    '''
    d3 = tf.layers.Dense(1,
                        kernel_initializer=init,
                        activation=tf.keras.activations.relu)
    ds =d3.apply(ds)
    '''

    y_p = ds-_y
    cost = tf.losses.mean_squared_error(_y,ds)
    opt = tf.train.GradientDescentOptimizer(0.05)
    #opt = tf.train.AdamOptimizer(0.1)

    grads_vars = opt.compute_gradients(cost)
    step = opt.apply_gradients(grads_vars)
    losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        kernels= (d.kernel.eval(),)#d2.kernel.eval())
        bias = (d.bias.eval(),)#d2.bias.eval())
        st = time.time()
        print bias
        for i in range(epochs):
            ker = d.kernel.eval()
            #print "ker \n%s"%str(ker)
            loss,yp,gv,_ = sess.run([cost,y_p,grads_vars,step],
                    feed_dict={_x:x,_y:y})
            print 'gv\n', np.max(gv[0][0])
            #hp.plot(gv[0][0].flatten())
            print "epoch #%i loss %f"%(i,loss)
            losses.append(loss)
            #print "errs\n %s\n"%str(yp)
    print  time.time() -st
    return kernels,bias,losses

def create_bike(x,y,bias=None,kernels=None,epochs=5):
    # Start my custom stuff
    d = Dense(x,1)
    d2 = Dense(d,1)
    d3 = Dense(d2,1)
    # Assign Kernel matrix to weights
    if kernels is not None:
        print 'k'
        d.W = kernels[0]
        #d2.W = kernels[1]
        d.b = bias[0]
        #d2.b = bias[1]
    relu = ReLu()
    sess = Session()
    sess.add_node(d)
    sess.add_node(ReLu())
    '''
    sess.add_node(d2)
    sess.add_node(ReLu())
    sess.add_node(d3)
    sess.add_node(relu)
    '''
    sess.add_node(subtr(d3,y))
    sess.add_node(MeanSqrError(d3))
    print '\n'
    return sess

def run_bike(x,y,sess,kernels=None,epochs=5):
    losses = []
    st=time.time()
    for i in range(epochs):
        '''
        x1 = d2.forward(x1)
        print 'shape:%s\n'%str(x1.shape),x1
        x1 =relu.forward(x1)
        print 'shape:%s\n'%str(x1.shape),x1
        '''
        #d.W = kernels[i]
        print "Weight matr "
        #print str(d.W)
        err=sess.step(x,y)
        #print "err na\n", err
        print 'epoch ',i,"LOSS:",err
        losses.append(err)
        print
    #hp.plot(d.W.flatten())

    print  time.time() -st
    return losses

epochs = 30
x,y = gen_wine()
x,xt=x[:1300],x[1300:]
y,yt=y[:1300],y[1300:]
kernels,bias,losses = run_tf(x,y,epochs=epochs)
plt.plot(range(epochs),losses,label='tf')
sess=create_bike(x,y,bias,kernels=kernels,epochs=epochs)
losses =run_bike(x,y,sess,epochs=epochs)
plt.plot(range(epochs),losses,label='mine')
plt.legend()
plt.show()
