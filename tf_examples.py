import tensorflow as tf
import keras.backend as K

#basic tensorflow examples
#reference: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples

hello=tf.constant("hello")

print(hello)

sess=tf.Session()

print(sess.run(hello))

#declare again the const
hello=tf.constant("hello")
print(hello)

#we see different name

hello=tf.constant("hello",name="h")
print(hello)

#we see h_0:0

hello=tf.constant("hello",name="h")
print(hello)

#we see the h_1:0

#now try name scope:
with tf.variable_scope("hello_world"):
    hello=tf.constant("hello",name="h")

print(hello)
#we will see variable scope will be part of its full name: hello_world/h:0

#so the name of a tf variable/constant depends on the variable scope where it is created

#we can create multiple variables, constants, placeholders under the same name scope, we will see _0, _1 attached to the name of  namespace
print("create two constants with the same given name under a variable scope")
with tf.variable_scope("hello_world"):  #re-open the scope, tensorflow memorize which time it is opened, wich is used as part of variables names
    print(tf.get_variable_scope().name) # hello_world
    hello=tf.constant("hello",name="h")
    print(hello) #hello_world_1/h:0
    hello=tf.constant("hello",name="h")
    print(hello)#hello_world_1/h_1:0

print("hiratical variable scope")
with tf.variable_scope("hello"):
    with tf.variable_scope("world"):
        hello=tf.constant("hello",name="h")
        print(hello) #we will see hello/world/h:0
        hello=tf.constant("hello",name="h") #we will see hello/world/h_1:0
        print(hello)
#conclusion: what you see is not what you get: depending on the variable scope and other variables under the scope  when it is created
#conclusion 2: once a variable is created, it has a unique name

#now let's study shariable variables
#basically, you can use "name" to get a varialble, you can image that tensor flow maintain a name space tree, each tree node has sub name spaces and variables as children
x=tf.Variable(tf.zeros([32]),name="x")   #the coding style: class name starting with a character in upper case, while function name in lower case; however naming convinence is not consistant, sometimes we see class name starting with a character in lower case, for example, constant, this is bad!
print(x.name)
with tf.variable_scope("temporal_variables"):
    x=tf.Variable(tf.zeros([32]),name="x")
    print(x.name) # temporal_variables/x:0
    x=tf.Variable(tf.zeros([32]),name="x")
    print(x.name) #temporal_variables/x_1:0

#shariable variables
with tf.variable_scope("sharable_variables"):
    x=tf.Variable(tf.zeros([32]),name="x") # there is no way to access non-sharable variables by get_variable
    y=tf.get_variable("x",shape=(32,),initializer=tf.constant_initializer(.0))#create a sharable variable
    print(x.name)
    print(y.name) #by default, variable sharing in this scope is not enabled, so get_variable will create a new sharable variable
    assert not  y == x
    #now enable variable sharing for this scope
    tf.get_variable_scope().reuse_variables()
    z=tf.get_variable("x",shape=(32,),initializer=tf.constant_initializer(.0)) #only search shariable variables directly under this scope
    print(z.name)
    assert not z==x #why variables declared with Variable is not re-usualble, only variables created by get_variable is sharable
    assert z==y
    #a variable is re-usable= scope reuse variables enabled AND variable itself is re-usable
    #again, name in get_variable depends on the calling context (scope, and non-shariable variables and sharable variables

#once created, refer to a variable by ref
print(z.name)
# now in the parent scope, its reuse property should not be affacted by the reuse property of any of its children
print( tf.get_variable_scope().reuse) #False

#can we re-use place holders? as indicated by the  name: get_variable, answer should be no
with tf.variable_scope("sharable_variables", reuse=True): # warning: this will actually not create  a new scope since reuse is set to true
    x=tf.placeholder(tf.float32,name='x')
    print(x.name) 
    try:
        y=tf.get_variable('x',shape=(),initializer=tf.constant_initializer(.0)) #this will try to get the sharable variable in the name space, yes, there is one, but with different shape, so we will get one exception
    except:
       assert True
    y = tf.get_variable("x",shape=(32,),initializer=tf.constant_initializer(.0))  # this time, okay, get the shariable variable x_1 (since the one with x is not shariable)
    print(y.name)   
#conclusion: variable_scope == get_variable_scope , if reuse=True, it will re-use existing name space
#place holders are not shariable, similarly for constants

with tf.variable_scope("sharable_variables"): #open an existing scope, but reuse is reset
    x2=tf.Variable(tf.zeros([32]),name="x")
    print("x2.name=%s"%(x2.name))
    try:
        x=tf.get_variable("x",shape=(32,),initializer=tf.constant_initializer(.0)) # bad since try to create a sharable variable which already exists in this scope
    except:
        assert True
    try:
        y=tf.get_variable("x",shape=(32,),initializer=tf.constant_initializer(.0)) # cannot create the same shariable variable
    except:
        assert True
    tf.get_variable_scope().reuse_variables()
    z=tf.get_variable("x",shape=(32,),initializer=tf.constant_initializer(.0)) # will be okay
    print(z.name)
    with tf.variable_scope("#1"):
        assert tf.get_variable_scope().reuse == True
        
    assert tf.get_variable_scope().reuse == True


'''
1. a shariable variable can only be created once
2. a shariable variable can be shared only if its scope allows sharing variables
3. in a scope enables sharing, get_variable will either creates a new  or reuses an existing shariable variable
4. in a scope that does not enable sharing, get_variable will creates a new shariable variable if it does not exist or throws an exception if it already exist
5. it is impossile to create two shariable variables with the same name
6. what you see a name is not what you get 

if we follow the principal that we should always avoid using  global variables, we should avoid using shariable variables since they are scope level shariable variables. I would like to say shariable variables are one of the worse design of tensorflow.
'''
#let's see one example, two developers dev two functions, foo1 and foo2, respectively
def foo1():
    with tf.variable_scope("same_scope"):
        x=tf.get_variable("x",shape=(32,),initializer=tf.constant_initializer(.0))
   #other operations

def foo2():
    with tf.variable_scope("same_scope"):     
        x=tf.get_variable("x",shape=(32,),initializer=tf.constant_initializer(.0)) #you would nevel know if this line works or not
   #other operations

foo1()
try:
    foo2()
except:
    assert True

#now let's play with devices: my expectation is that it only specifies which ops should run in which device, with nothing to do with scope
#try to pin variable into some device
with tf.device("/cpu:0"):
    count=tf.Variable(0) # a variable must be initialized before it is used
    sess.run(count.initializer)
    print(count.name)
count = count + 1
print(sess.run(count))

def foo3():
    v=tf.Variable(0)
    sess.run(v.initializer) #keras provides a good wraper variable(initial_values)
    return v+1

with tf.device("/cpu:0"):
    ret=foo3()
    print(sess.run(ret))
