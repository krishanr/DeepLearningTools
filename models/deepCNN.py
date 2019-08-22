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

def deep_cnn_model(X_train, y_train, X_val, y_val, params, verbose = 1):
    """A deep convolutional neural network designed for the CIFAR10 dataset. It creates and fits a model using details in params.
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
    
    reg_param = 0 if (not params.get('reg_param')) else params['reg_param']
    dropout_prob = 0 if (not params.get('dropout_prob')) else params['dropout_prob'] 
    callbacks = [] if (not params.get('callbacks')) else params['callbacks']

    #Chose the optimizer
    optimizer = params.get('optimizer')
    if optimizer == 'adagrad':
        optimizer = Adagrad
    elif optimizer == 'nadam':
        optimizer = NAdam
    elif optimizer == 'adadelta':
        optimizer = Adadelta
    else:
        optimizer = Adam

    #Set default values for the final layers.
    if params.get('layers'):
        lds = params['layers']
    else:
        lds = [512, 128]
        params['layers'] = lds

    params['actCounter'] = 0
    def act(x = None):
        if x == None:
            res = PReLU() if params['activation'] == 'prelu' else Activation(params['activation'], name='act'+ str(params['actCounter']))
        else:
            res = PReLU()(x) if params['activation'] == 'prelu' else Activation(params['activation'], name='act'+ str(params['actCounter']))(x)
        params['actCounter'] += 1
        return res

    X_input = Input(X_train[0].shape)
    #X = Dropout(dropout_prob)(X_input)

    #First convolutional block.
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(32, (4, 4), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = act(X)
    X = MaxPooling2D((3, 3), strides = 2, name='max_pool0')(X)
    if dropout_prob > 0:
        X = Dropout(dropout_prob)(X)

    #Second convolutional block.
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = act(X)
    X = MaxPooling2D((3, 3), strides = 2, name='max_pool1')(X)
    if dropout_prob > 0:
        X = Dropout(dropout_prob)(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    for i,l in enumerate(lds):
        X = Dense(l, activation='relu', name='fc' + str(i+1),
        kernel_initializer=keras.initializers.he_uniform(),
        kernel_regularizer=regularizers.l2(reg_param) )(X)
        X = BatchNormalization(name = 'bn' + str(i+2))(X)
        if dropout_prob > 0:
            X = Dropout(dropout_prob)(X)

    X = Dense(params['classes'], activation='softmax', name='fc0')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='deepCNNModel')
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