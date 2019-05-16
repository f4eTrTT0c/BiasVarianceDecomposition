# -*- coding: utf-8 -*-
"""
Train a specified classifier for increasing levels of complexity.
"""
import sys
import os
import numpy as np
import load_data
from bootstrap import Bootstrapper

# ============================================================================
# Pick 'RandomForestClassifier', 'MLP', or 'CNN'
used_classifier = 'RandomForestClassifier'      # classifier to use

num_classes = load_data.n_classes()
experiment_dir = 'complexity/'
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

# ============================================================================
# Load classifier settings
if used_classifier == 'RandomForestClassifier':
    num_bootstraps = 100
    complexity_numbers = [5, 10, 25, 50, 100, 250, 500]
    classifier_name = 'RF'

    from train_helper import RF_model_save_builder as model_saver
    complexity_indicator = 'num_trees'
    predict_proba_params = {}
elif used_classifier == 'MLP':
    num_bootstraps = 10
    complexity_numbers = list(range(0, 6))
    num_epochs = 50
    batch_size = 64
    loss_function = 'mean_squared_error'

    classifier_name = 'MLP_{}'.format(loss_function)

    from train_helper_DL import MLP_model_save_builder as model_saver
    complexity_indicator = 'num_hidden_layers'
    predict_proba_params = {'num_classes': num_classes,
                            'num_epochs': num_epochs,
                            'loss_function': loss_function,
                            'batch_size': batch_size}
elif used_classifier == 'CNN':
    num_bootstraps = 10
    num_epochs = 50
    batch_size = 64
    loss_function = 'mean_squared_error'
    BN_in_block = True
    max_pooling_in_block = True

    # If there is no max pooling, these experiments last too long for more than
    # 3 layers, as the number of parameters grow exponentially
    if max_pooling_in_block:
        complexity_numbers = list(range(0, 6))
    else:
        complexity_numbers = list(range(0, 4))

    classifier_name = 'CNN_{}_mp{}_bn{}'.format(
            loss_function, max_pooling_in_block, BN_in_block)

    from train_helper_DL import CNN_model_save_builder as model_saver
    complexity_indicator = 'num_convblocks'
    predict_proba_params = {'num_classes': num_classes,
                            'num_epochs': num_epochs,
                            'loss_function': loss_function,
                            'batch_size': batch_size,
                            'BN_in_block': BN_in_block,
                            'max_pooling_in_block': max_pooling_in_block}
else:
    sys.exit('Unknown classifier, please select '
             'RandomForestClassifier, '
             'MLP, or '
             'CNN')

# ============================================================================
# Load dataset
if used_classifier in ['RandomForestClassifier', 'MLP']:
    (x_train, y_train, x_validate, y_validate,
     x_test, y_test) = load_data.load_data_selected()
else:
    (x_train, y_train, x_validate, y_validate,
     x_test, y_test) = load_data.load_data()

num_features = x_train.shape[1]

if used_classifier == 'CNN':
    # Reshape x to be readable by the CNN
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_validate = x_validate.reshape((
            x_validate.shape[0], x_validate.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))


# ============================================================================
# Derive identifier (filenames), number of classes, load data
identifier = '{}_'.format(classifier_name)
variable_list = complexity_numbers
variable_list_length = len(variable_list)


# ============================================================================
# Use Bootstrapper to create random datasets and run the predict_proba function
bootstrapper = Bootstrapper(num_classes, x_train, y_train,
                            x_validate, y_validate, x_test, y_test)


# ============================================================================
# Load status (so when a task times out/fails, no work is redone)
if os.path.isfile(identifier + 'status.npy'):
    status = np.load(identifier + 'status.npy').astype(int)[0]
else:
    status = 0


# ============================================================================
# For an increasing number of training samples, save predict_probas
for variable_list_index in range(status, variable_list_length):
    variable_current = variable_list[variable_list_index]
    print('Starting iteration {} with parameter={}'.format(
            variable_list_index, variable_current))

    locstring = (experiment_dir + identifier +
                 'complexity{complexity}_run{i_bootstrap}')
    run_and_save = model_saver(locstring)

    # Put in the complexity as an argument
    predict_proba_params_update = {**predict_proba_params,
                                   complexity_indicator: variable_current}

    accuracy = bootstrapper.bootstrap_run(
            run_and_save, num_bootstraps,
            locpars={'complexity': variable_current},
            **predict_proba_params_update)

    print('Accuracy score(s): {}'.format(accuracy))

    np.save(identifier + 'status.npy', [variable_list_index+1])

print('Finished succesfully')
