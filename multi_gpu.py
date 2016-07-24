import tensorflow as tf
import numpy as np
import datetime
#reference: https://www.tensorflow.org/versions/r0.9/how_tos/using_gpu/index.html
#reference:https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_MultiGPU/multigpu_basics.py

A=np.random.random((10,10))
B=np.random.random((10,10))
a=tf.constant(A,name='a')
b=tf.constant(B,name='b')

def foo(a,b):
    return tf.matmul(a,b)

sess=tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True))

c=foo(a,b)


#manual placement
with tf.device('/gpu:0'):
    c1=foo(a,b)

sess.run([c,c1]) #this will require to place matmul on gpu:0, and other ops on cpu:0; tensorflow will take care of data transferring between cpu and gpu 

