# -*- coding: utf-8 -*-
"""
Train a specified classifier for increasing numbers of features
"""
import sys
import os
import numpy as np
import load_data
from bootstrap import Bootstrapper

# ============================================================================
# Pick 'RandomForestClassifier', 'MLP'
used_classifier = 'RandomForestClassifier'      # classifier to use

num_classes = load_data.n_classes()
experiment_dir = 'numfeatures/'
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

# ============================================================================
# Load classifier settings
if used_classifier == 'RandomForestClassifier':
    num_bootstraps = 100
    num_trees = 100
    classifier_name = 'RF{}'.format(num_trees)

    from train_helper import RF_model_save_builder as model_saver
    predict_proba_params = {'num_trees': num_trees}
elif used_classifier == 'MLP':
    num_bootstraps = 10
    num_hidden_layers = 2
    num_epochs = 50
    batch_size = 64
    loss_function = 'mean_squared_error'

    classifier_name = 'MLP{}_{}'.format(num_hidden_layers, loss_function)

    from train_helper_DL import MLP_model_save_builder as model_saver
    predict_proba_params = {'num_hidden_layers': num_hidden_layers,
                            'num_classes': num_classes,
                            'num_epochs': num_epochs,
                            'loss_function': loss_function,
                            'batch_size': batch_size}
else:
    sys.exit('Unknown classifier, please select '
             'RandomForestClassifier, or '
             'MLP')


# ============================================================================
# Settings: #features to use in the training set as training data
feature_numbers = [25, 50, 75, 100, 125, 150, 175, 200]


# ============================================================================
# Derive identifier (filenames), number of classes, load data
identifier = '{}_'.format(classifier_name)
variable_list = feature_numbers
variable_list_length = len(variable_list)


# ============================================================================
# Load dataset
(x_train, y_train, x_validate, y_validate,
 x_test, y_test) = load_data.load_data_selected()

num_features = x_train.shape[1]


# ============================================================================
# Use Bootstrapper to create random datasets and run the predict_proba function
bootstrapper = Bootstrapper(num_classes, x_train, y_train,
                            x_validate, y_validate, x_test, y_test)


# ============================================================================
# Load status
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
                 '{num_features}features_run{i_bootstrap}')
    run_and_save = model_saver(locstring)

    accuracy = bootstrapper.bootstrap_run(
            run_and_save, num_bootstraps,
            n_features=variable_current,
            locpars={'num_features': variable_current},
            **predict_proba_params)

    print('Accuracy score(s): {}'.format(accuracy))

    np.save(identifier + 'status.npy', [variable_list_index+1])

print('Finished succesfully')
