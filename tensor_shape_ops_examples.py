import tensorflow as tf
from tensorflow import  Variable as variable
from  tensorflow import  placeholder
from  tensorflow import  constant
import numpy as np
#default global session
session=tf.Session()
#None means any length, -1 means all remaining length
x=placeholder(tf.float32,shape=(None,None))
print(tf.shape(x)) #shape is a ID tensor of int32
print(x.get_shape()) #tensor shape, a tuple
print(x)
ndim=len(x.get_shape())
print(ndim)

y=placeholder(tf.float32,shape=(None,None))
z=x+y
print(tf.shape(z))
print(z.get_shape())
#some ops that related to shape

x1=tf.reshape(x,shape=(-1,))
print(x1.get_shape())
x2=tf.reshape(x1,shape=(-1,3)) #reshape requires int, None will be bad
print(x2.get_shape())
x3=tf.reshape(x1,shape=tf.shape(x)) #reshape could be a shape tensor (ID tensor of int)
print(x3.get_shape())
print(tf.shape(x)[0])

w=placeholder(tf.float32,shape=(2,3))
#expand/squeeze dim
assert tf.expand_dims(w,0).get_shape() == (1,2,3)
assert tf.expand_dims(w,1).get_shape() == (2,1,3)
assert tf.expand_dims(w,-1).get_shape() == (2,3,1)

w11=tf.expand_dims(tf.expand_dims(w,0),0)
assert tf.squeeze(w11,[0,1]).get_shape() == (2,3)

#permutate dimensions
assert tf.transpose(w,(1,0)).get_shape() == (3,2) # pattern is a tuple of list, both okay

#concat and split
x1=placeholder(tf.float32,shape=(2,3))
x2=placeholder(tf.float32,shape=(2,6))
assert tf.concat(1,values=[x1,x2]).get_shape() == (2,9) #not consistent, sometimes, -1 means the last dim as in expand_dims, some times, you have to use the positive last dim, keras provides consistent APIs
x_list=tf.split(split_dim=1,num_split=3,value=x2)
assert len(x_list) == 3
for _ in x_list:
    assert _.get_shape() == (2,2)
#slice
x=placeholder(tf.float32,shape=(10,5,20))
x1=tf.slice(x,begin=(2,0,0),size=(1,-1,-1)) # the third sample, -1 means all the remaining elements along that dimension
assert x1.get_shape() == (1,5,20)
x2=tf.slice(x,begin=(2,3,10), size=(1,1,-1)) # similar to x[begin[0]:begin[0]+size[0], begin[1]:begin[1]+size[1],...]
assert x2.get_shape() == (1,1,10)
#pack and unpack: increase/decrease dimensions
x=placeholder(tf.float32,shape=(10,5,20))
x_list=tf.unpack(x)
assert len(x_list) == 10
for _ in x_list:
    assert(_.get_shape() == (5,20))
x2=tf.pack(x_list)
assert x2.get_shape() == (10,5,20)

#gather: use indices to select elements from a tensor
#output[i]=input[indices[i]], considering input as a list of tensors with a  shape of  shape[1:]
#output[i,j,...]=input[indices[i,j,...],:,...:]
x=placeholder(tf.float32, shape=(10,5,20))
indices = placeholder(tf.int32,shape=(20,)) 
y=tf.gather(params=x,indices=indices) #get 20 of tensors (of shape (5,20)) from x 
assert y.get_shape() == (20,5,20)
#another version: gather_nd, each indice's last dimension has a rank R, where R is the dimensions of input, meaning that a indices holding a indice for each dimension
#output[i,j,...]=input[indices[i,j,...,:]]
indices=placeholder(tf.int32,shape=(100,3)) #select 100 elements from x
y=tf.gather_nd(params=x,indices=indices)
assert y.get_shape() == (100,)

#more examples: see keras, e.g., repeat_elements

def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat

    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    '''
    x_shape = x.get_shape().as_list()
    # slices along the repeat axis
    splits = tf.split(axis, x_shape[axis], x)
    # repeat each slice the given number of reps
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis, x_rep)

#https://www.tensorflow.org/versions/r0.9/api_docs/python/array_ops.html#shapes-and-shaping 
