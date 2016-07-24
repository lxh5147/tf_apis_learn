import tensorflow as tf
import numpy as np

__global_session = None 
def get_session():
    global __global_session
    if __global_session is None:
        __global_session = tf.Session()
    return __global_session
#basic operations on a variable
def variable(value, dtype='float32',  name=None):
    v=tf.Variable(np.asarray(value,dtype=dtype),name=name)
    get_session().run(v.initializer)
    return v
weights = variable(np.random.random((2,3)))
tf.assign(weights, weights*2)
print(get_session().run(weights))

#batch initialization
weights=tf.Variable(tf.random_normal([784,200],stddev=0.35))
init = tf.initialize_all_variables()
get_session().run(init)
weights += .5
print(get_session().run(weights))

'''
refer to the constructor to get all possible construction parameters of a variable, such as trainable, validate_shape, etc
tf.Variable.__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None)
'''

#re-use initial value -- important to ensure two variables have the same value
v = tf.Variable(tf.truncated_normal([2, 3]))
# Use `initialized_value` to guarantee that `v` has been
# initialized before its value is used to initialize `w`.
# The random values are picked only once.
w = tf.Variable(v.initialized_value() * 2.0)

init = tf.initialize_all_variables()
get_session().run(init)

print(get_session().run(v))
print(get_session().run(w))

