import numpy as np

import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers.advanced_activations import PReLU 
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras import regularizers
from keras.optimizers import Adam, Nadam, Adagrad, Adadelta

import time

def vgg_model(X_train, y_train, X_val, y_val, params, verbose = 1):
    """An abstraction of the standard VGG16 convolutional neural network. It creates and fits a model using details in params. This module
       uses params['layers'] to create a variable number of convolutional blocks with increasing layer size and a variable number of fully
       connected layers which follow the convolutional blocks.
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
    
    reg_param = params.get('reg_param', 0)
    dropout_prob = params.get('dropout_prob', 0)
    callbacks = params.get('callbacks', [])

    #Chose the optimizer
    optimizer = params.get('optimizer', 'adam')
    if optimizer == 'adagrad':
        optimizer = Adagrad
    elif optimizer == 'nadam':
        optimizer = NAdam
    elif optimizer == 'adadelta':
        optimizer = Adadelta
    else:
        optimizer = Adam

    lds = params.get('layers', [3, 512])
    params['actCounter'] = 0
    def act(x = None):
        if x == None:
            res = PReLU() if params['activation'] == 'prelu' else Activation(params['activation'], name='act'+ str(params['actCounter']))
        else:
            res = PReLU()(x) if params['activation'] == 'prelu' else Activation(params['activation'], name='act'+ str(params['actCounter']))(x)
        params['actCounter'] += 1
        return res

    X_input = Input(X_train[0].shape)
    
    #Initalize the convolutional block parameters.
    convF = params.get('conv_filter', 3)
    convLBase = params.get('conv_layer_start', 32)
    poolSize = params.get('pool_size', 2)
    poolStride = params.get('pool_stride', 2)
    for i in range(0, lds[0]):
        X = Conv2D(convLBase * (i+1), (convF, convF), padding='same', name = 'conv' + str(i))(X_input if i == 0 else X)
        X = BatchNormalization(axis = 3, name = 'bn0' + str(i))(X)
        X = act(X)
        X = Conv2D(convLBase * (i+1), (convF, convF), name = 'conv' + str(i) + '0')(X)
        X = BatchNormalization(axis = 3, name = 'bn1' + str(i))(X)
        X = act(X)
        X = MaxPooling2D((poolSize, poolSize), strides = poolStride, name='max_pool' + str(i))(X)
        if dropout_prob > 0:
            X = Dropout(dropout_prob)(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    for i,l in enumerate(lds[1:]):
        X = Dense(l, activation='relu', name='fc' + str(i+1),
        kernel_initializer=keras.initializers.he_uniform(),
        kernel_regularizer=regularizers.l2(reg_param) )(X)
        X = BatchNormalization(name = 'bn2' + str(i))(X)
        if dropout_prob > 0:
            X = Dropout(dropout_prob)(X)

    X = Dense(params.get('classes', 2), activation='softmax', name='fc0')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='VGG')
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