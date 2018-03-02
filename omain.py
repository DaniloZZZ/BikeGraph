from ReLu import ReLu,Square
from Dense import Dense
from Session import Session

import numpy as np
import hello
import time,os
import tensorflow as tf
import hipsterplot as hp
import matplotlib.pyplot as plt


# defining a data generation functions
def gen_reg():
    x = np.random.rand(200,45)
    y = .3*x[:,0]+x[:,2]+x[:,3]
    return x,y.reshape(200,1)

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
    fnum = 45
    _x = tf.placeholder(tf.float32, shape=(None,fnum), name='data')
    _y = tf.placeholder(tf.float32, shape=(None,1), name='labels')
    d = tf.layers.Dense(3,
                        use_bias=False,
                        activation=tf.keras.activations.relu)
    ds =d.apply(_x)
    ds = tf.layers.dense(ds,1,use_bias=False,
                    activation=tf.keras.activations.relu)
    '''
    tf.assign(d.kernel,tf.Variable(weg.T.astype(np.float32)))
    '''
    y_p = ds-_y
    cost = tf.losses.mean_squared_error(_y,ds)
    opt = tf.train.GradientDescentOptimizer(0.03)
    #opt = tf.train.AdamOptimizer(0.05)

    grads_vars = opt.compute_gradients(cost)
    step = opt.apply_gradients(grads_vars)
    kernels = []
    losses = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ker = d.kernel.eval()
        for i in range(epochs):
            ker = d.kernel.eval()
            #print "ker \n%s"%str(ker)
            loss,yp,gv,_ = sess.run([cost,y_p,grads_vars,step],
                    feed_dict={_x:x,_y:y})
            kernels.append(ker)
            print 'gv\n', np.max(gv[0][0])
            #hp.plot(gv[0][0].flatten())
            print "epoch #%i loss %f"%(i,loss)
            losses.append(loss)
            #print "errs\n %s\n"%str(yp)
    return kernels,losses

def run_bike(x,y,kernels=None,epochs=5):
    # Start my custom stuff
    d = Dense(x,3)
    d2 = Dense(d,1)
    # Assign Kernel matrix to weights
    if kernels:
        d.W = kernels[0]
    weg = d.W
    relu = ReLu()
    sq = Square()
    sess = Session()
    sess.add_node(d)
    sess.add_node(d2)
    sess.add_node(relu)
    print '\n'

    losses = []
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
        loss = np.mean(np.square(err))
        print 'epoch ',i,"LOSS:",loss
        losses.append(loss)
        print
    hp.plot(d.W.flatten())

    return losses

epochs = 4
x,y = gen_reg()
kernels,losses = run_tf(x,y,epochs=epochs)
plt.plot(range(epochs),losses,label='tf')
losses =run_bike(x,y,kernels,epochs=epochs)
plt.plot(range(epochs),losses,label='mine')
plt.legend()
plt.show()
