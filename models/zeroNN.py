import keras
from keras import layers
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras import regularizers
from keras.optimizers import RMSprop, Adam, Nadam, Adagrad, Adadelta

import time


def zero_nn_model(X_train, y_train, X_val, y_val, params, verbose = 1):
    """A simple zero layer neural network is created and fitted using the model details in params.

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

    #Chose the optimizer
    optimizer = params.get('optimizer')
    if optimizer == 'adagrad':
        optimizer = Adagrad
    elif optimizer == 'nadam':
        optimizer = Nadam
    elif optimizer == 'adadelta':
        optimizer = Adadelta
    elif optimizer == 'rmsprop':
        optimizer = RMSprop
    else:
        optimizer = Adam

    reg_param = 0 if (not params.get('reg_param')) else params['reg_param']
    callbacks = [] if (not params.get('callbacks')) else params['callbacks']
    
    X_input = Input(X_train[0].shape)

    X = Dense(params['classes'], activation=params['activation'], name='fc0',
        kernel_regularizer=regularizers.l2(reg_param) )(X_input)

    # Create model.
    model = Model(inputs = X_input, outputs = X, name='ZNNModel')
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