import keras
from keras import layers
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras import regularizers
from keras.optimizers import Adam, Nadam, Adagrad, Adadelta

import time


def deep_nn_model(X_train, y_train, X_val, y_val, params, verbose = 1):
    """A generic multi-layer neural network is created and fitted using the model details in params.

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
    
    #Set default values fo various parameters.
    if params.get('layers'):
        lds = params['layers']
    else:
        lds = [1000, 500, 500, 500]
        params['layers'] = lds

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

    reg_param = 0 if (not params.get('reg_param')) else params['reg_param']
    dropout_prob = 0 if (not params.get('dropout_prob')) else params['dropout_prob'] 
    callbacks = [] if (not params.get('callbacks')) else params['callbacks']
    
    X_input = Input(X_train[0].shape)
    X = Dense(lds[0], activation= params['activation'], name='fc0',
        kernel_initializer=keras.initializers.he_uniform(),
        kernel_regularizer=regularizers.l2(reg_param) )(X_input)
    if dropout_prob > 0:
        X = Dropout(dropout_prob)(X)

    for i,l in enumerate(lds[1:-1]):
        X = Dense(l, activation=params['activation'], name='fc' + str(i+1),
        kernel_initializer=keras.initializers.he_uniform(),
        kernel_regularizer=regularizers.l2(reg_param) )(X)
        if dropout_prob > 0:
            X = Dropout(dropout_prob)(X)

    X = Dense(params['classes'], activation='softmax', name='fc' + str(len(lds)),
        kernel_initializer=keras.initializers.he_uniform(),
        kernel_regularizer=regularizers.l2(reg_param) )(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='DNNModel')
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