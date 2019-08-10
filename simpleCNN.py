import numpy as np

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers.advanced_activations import PReLU 
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam, Nadam, Adagrad, Adadelta

import time


def simple_cnn_model(X_train, y_train, X_val, y_val, params, verbose = 1):
    """A simple single stage convolutional neural network is created and fitted using the model details in params.
       This model was taken from an assignment in Andrew Ng's course on convolutional neural networks.
    Arguments:
    X_train -- The training data.
    y_train -- The label vector, with integer values.
    X_val -- The validation data.
    y_val -- The label vector for the validation data.
    params -- A dictionary of parametes for the model such as activation, layers etc.
    verbose -- 1 means show the output from the call to fit.

    Returns:
    out -- Output from the call to keras model.fit.
    model -- a Model() instance in keras
    t_diff -- The time taken for the call to keras model.fit.
    """

    dropout_prob = 0 if (not params.get('dropout_prob')) else params['dropout_prob'] 
    callbacks = [] if (not params.get('callbacks')) else params['callbacks']

    #Choose the optimizer
    optimizer = params.get('optimizer')
    if optimizer == 'adagrad':
        optimizer = Adagrad
    elif optimizer == 'nadam':
        optimizer = NAdam
    elif optimizer == 'adadelta':
        optimizer = Adadelta
    else:
        optimizer = Adam

    X_input = Input(X_train[0].shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Dropout(dropout_prob)(X)
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    if params['activation'] == 'prelu':
        X = PReLU()(X)
    if not (params['activation'] == 'prelu'):
        X = Activation(params['activation'])(X)
    X = Dropout(dropout_prob)(X)
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(params['classes'], activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='SimpleCNNModel')
    model.compile(optimizer=optimizer(lr=params['lr']),
                  loss=params['loss'],
                  metrics=['accuracy'])

    t_start = time.clock()
    out = model.fit(X_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=verbose,
                    validation_data= [X_val, y_val],
                    callbacks = callbacks)
    t_end = time.clock()
    t_diff = t_end - t_start

    return out, model, t_diff
