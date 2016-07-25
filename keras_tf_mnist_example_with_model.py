import tensorflow as tf
import keras.backend as K
import numpy as np

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
K.set_session(sess)

#happy journey started!

from keras.layers import Input,Dense,Dropout,BatchNormalization
from keras.models import Model

def build_model():
    img=Input(shape=(784,))
    x=Dense(128,activation='relu')(img)
    x=Dropout(0.5)(x)
    x=BatchNormalization()(x)
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.5)(x)   
    preds=Dense(10,activation='softmax')(x)
    model = Model(input=img, output=preds)
    return model


model=build_model()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

model.fit(mnist_data.train.images,mnist_data.train.labels,batch_size=50,nb_epoch=10,validation_split=.1)

model.evaluate(mnist_data.test.images,mnist_data.test.labels,batch_size=50)




