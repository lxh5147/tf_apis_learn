import tensorflow as tf
import keras.backend as K
import numpy as np

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
K.set_session(sess)

#happy journey started!

from keras.layers import Input,Dense,Dropout
img=Input(shape=(784,))
x=Dense(128,activation='relu')(img)
x=Dropout(0.5)(x)
x=Dense(128,activation='relu')(x)
x=Dropout(0.5)(x)

preds=Dense(10,activation='softmax')(x)
labels=Input(shape=(10,))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds,labels=labels))

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

for _ in range(10):
    batch=mnist_data.train.next_batch(50)
    sess.run(train_step,feed_dict={img:batch[0],labels:batch[1],K.learning_phase():1})

from keras.metrics import categorical_accuracy as accuracy

acc_value=accuracy(labels,preds)

print (sess.run(acc_value,feed_dict={img:mnist_data.test.images,labels:mnist_data.test.labels,K.learning_phase():0}))



