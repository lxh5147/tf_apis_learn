import tensorflow as tf
import numpy as np

__global_session = None 
def get_session():
    global __global_session
    if __global_session is None:
        __global_session = tf.Session()
    return __global_session

#place holder is like constant tensor, but you have to  initialize it every time when run a graph by providing a value through feed_dict

x=tf.placeholder(tf.float32) # place_holder vs. Variable --> inconsistent naming style
y=tf.placeholder(tf.float32)
z=x+y
w=x-y
print( get_session().run([z,w],feed_dict={x:0.6,y:.2}) )

