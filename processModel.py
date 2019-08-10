"""
Note: This class has only been tested with the following
versions of keras, hyperopt and networkx:
pip install keras==2.1.6 hyperopt==0.1
pip install networkx==1.11

A special thanks to Vik for giving me the idea to create this class for running
deep learning experiments.
"""

import csv
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
import numpy as np

from keras import backend as K


class MetaModel():
    """This class exposes its primary function, process_model, which trains multiple models generated by
    various combinations of parameters and optimized hyper parameters. The hyper parameters are given to hyperopt to optimize,
    then the trained model is saved, and its details such as test set accuracy are saved in the self.model_res_file csv file.
    Notes: Models are saved in local_dir + '/trained_models/' after they are trained.
    """

    def __init__(self, local_dir, model_res_file = 'modelResults.csv',
                 verbose=1, max_evals= 3):
        """Arguments:
                     local_dir -- The file system directory to save various results to.
                     model_res_file -- The csv file (relative to local_dir) where various attributes of the fitted model,
                     such as the loss, will be saved.
                     verbose -- 1 means the output will be printed.
                     max_evals -- Maximum number of evaluations done by the fmin function of hyperopt.
           Returns: 
                    None
        """
        self.local_dir = local_dir
        self.model_res_file = model_res_file
        self.verbose = verbose
        self.max_evals = max_evals

        self.model_results = []

    def generate_model_name(self, model_name, params, attributes):
        """ Method to generate the file name for the model with given model_name, params and attributes.
        """
        generated_name = model_name
        if 'layers' in params:
            generated_name += '-' + ",".join([str(x) for x in params['layers']])
        if ('optimizer' in params) and isinstance(params['optimizer'], str):
            generated_name += '-' + params['optimizer']			
        if 'activation' in params and isinstance(params['activation'], str):
            generated_name += '-' + params['activation']		
        if 'normTech' in attributes and isinstance(attributes['normTech'], str):
            generated_name += '-' + attributes['normTech']
        return generated_name

    def eval_model2(self, params, hyper_params, model_generator, model_name, train_test, attributes):
        """This method will fit the model returned my model_generator on train_test data by first optimizing
           the hyper_params using hyperopt. The best model returned by hyperopt will be saved to a file whose name
           is generated using model_name, and then various attributes about this model specified by the attributes
           variable will be saved to self.model_result_file.

            Only arguments which differ from process_model will be described.
            Arguments:
                     params -- A dictionary of model parameters to be sent to model_generator for execution.
           Returns: 
                    None
        """

        X_train, X_val, X_test, y_train, y_val, y_test = train_test  

        out = None
        fit_time = None
        #Define the function we'd like hyperopt to optimize.
        #Its input are the optimization parameters.
        def hyperopt_objective(x):
            if self.verbose:
                print({**params, **x})
            __out, my_model, __fit_time = model_generator(X_train, y_train, X_val, y_val, {**params, **x}, verbose = self.verbose)
            #Store these variables for later use.
            out = __out
            fit_time = __fit_time
            #Generate a score on the validation data.
            val_acc = my_model.evaluate(X_val, y_val, verbose = self.verbose)[0]
            return {'loss':  val_acc, 'status': STATUS_OK, 'fit_time': fit_time, 'model' : my_model, 'out' : out}

        #We will train a model based on the best parameters given by hyperopt
        trials = Trials()
        best = fmin(hyperopt_objective, hyper_params, 
                    algo=tpe.suggest, max_evals=self.max_evals, trials= trials)

        best_trial = None
        #Find the best model.
        for trial in trials.trials:
            if best_trial == None:
                best_trial = trial
            elif best_trial['result']['loss'] > trial['result']['loss']:
                best_trial = trial			
            #Otherwise preserve the best_trial.
        model = best_trial['result']['model']
        out = best_trial['result']['out']
        fit_time = best_trial['result']['fit_time']

        #Save the model for later use.
        model.save(self.local_dir + '/trained_models/' + self.generate_model_name(model_name, params, attributes) + '.h5')
        self.model_results.append([out, best_trial, trials])
        
        #Evaluate the model.
        train_score = model.evaluate(X_train, y_train, verbose = self.verbose)
        test_score = model.evaluate(X_test, y_test, verbose = self.verbose)
        if self.verbose:
            print ("Loss = " + str(test_score[0]))
            print ("Test Accuracy = " + str(test_score[1]))
        with open(self.local_dir + '/' + self.model_res_file, mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #Merge params with tuned hyper_params for recording results.
            params = {**params, **best}
            #Save attribute values.
            my_attributes = attributes.copy()
            my_attributes[1] = train_score[1]
            my_attributes[2] = fit_time
            my_attributes[3] = test_score[1]
            #Here we write params/hyper_params specified in attributes to my_attributes before writing
            #my_attributes to the model_res_file csv.
            if (len(attributes) > 4):
                for i in range(4,len(attributes)):
                    if isinstance(attributes[i], str) and (attributes[i] in params):
                        if attributes[i] == 'layers':
                            my_attributes[i] = "-".join([str(x) for x in params['layers']])
                        else:
                            #We'll search for this attribute in the parameters and record the parameter value.
                            my_attributes[i] = str(params[attributes[i]])
                    else:
                        my_attributes[i] = np.NaN
            writer.writerow(my_attributes)

    def process_model(self, params, hyper_params, model_generator, model_name, train_test, attributes):
        """Arguments:
                     params -- A dictionary of lists where the key is a model parameter such as activation, and the list is its possible values
                               such as [relu, sigmoid].
                     hyper_params -- A dictionary containing hyperopt optimization parameters. See hyoperopt for details.
                     model_generator -- A function returning a fitted keras model among other things. Please see examples.
                     model_name -- A name for the model which is later used when saving the keras model to a file. E.g. SimpleNN.
                     train_test -- A list of the following form: [X_train, X_val, X_test, y_train, y_val, y_test].
                     attributes -- A list where most entries are strings which are keys into the params or hyper_params dictionaries
                                   that you wish to write in the self.model_res_file.
           Returns: 
                    self.model_results -- List[List[out, best_trial, trials]] where out is the best model output, best_trial is best trial
                    dictionary, and trials is a list of trial objects.
        """
        self.model_results = []
        self.recurse_params({}, params, hyper_params, model_generator, model_name, train_test, attributes)
        return self.model_results

    def recurse_params(self, path, params, hyper_params,
                        model_generator, model_name, train_test, attributes):
        """The purpose of this method is to unfold the dictionary of lists, params, and call eval_model2
           on each combination of params.
        """
        if (len(params) == 0):
            #End of the parameters, so evaluate these parameters.
            self.eval_model2(path, hyper_params, model_generator, model_name, train_test, attributes)
        else:
            key, vals = list(params.items())[0]
            for val in vals:
                new_path = path.copy()
                new_path[key] = val
                new_params = params.copy()
                del new_params[key]
                self.recurse_params(new_path, new_params, hyper_params,
                                    model_generator, model_name, train_test, attributes)
