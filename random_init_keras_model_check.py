
import os
# to use theano as the backend
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np

def build_simple_model(random_seed):

    np.random.seed(random_seed)

    from keras.models import Sequential
    from keras.layers import Dense, Activation

    # this will build all layers of this model. building a layer will initialize all parameters of this layer

    model = Sequential([
        Dense(6, input_shape=(5,)),
        Activation('relu'),
        Dense(8),
        Activation('softmax'),
    ])

    return model

def dump_model_parameters(model):
    # now show the values of the initial values of all parameters

    assert model
    import keras.backend as K

    print K.batch_get_value(model.weights)


if __name__ == '__main__':
    model = build_simple_model( random_seed=1234)
    dump_model_parameters(model)
    print '======'
    model = build_simple_model( random_seed=5678)
    dump_model_parameters(model)

