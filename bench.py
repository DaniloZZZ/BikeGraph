from ReLu import ReLu
import numpy as np
import hello
import time,os
import tensorflow as tf
import matplotlib.pyplot as plt

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
relu = ReLu()

x = np.linspace(-10,10,100)
plt.plot(x,relu.forward(x))
#plt.show()
b = 2
print 'fin', hello.baz(12)
print b
print 'matr'
sess = tf.InteractiveSession()

a = np.random.rand(100,120003).astype(dtype=np.float32)
b = np.random.rand(120003,1403).astype(dtype=np.float32)
print a.shape,b.shape

c = tf.matmul(tf.constant(a),tf.constant(b))
st = time.time()
c.eval()
print 'tf',time.time()-st

st = time.time()
z = np.dot(a,b)
print 'numpy',time.time()-st

print a.T.flags['F_CONTIGUOUS']
a = np.ascontiguousarray(a.T).T
b = np.ascontiguousarray(b.T).T
print a.flags['F_CONTIGUOUS']

st = time.time()
z = hello.dot(a,b)
print 'libblas',time.time()-st
