# -*- coding: utf-8 -*-
"""
Get predictions and guessing entropies from saved classifiers
of different complexities
"""
import os
import numpy as np
import pickle
import load_data
from sidechannel_tools import SideChannelEvaluator as SCE
from sidechannel_tools import PredictionsConverter as PC


# Directory for this experiment (read classifiers, write predictions/GEs)
experiment_dir = 'complexity/'


# classifier_name: (extension, n_runs, variables)
classifiers = {
        'RF': ('pkl', 100, [5, 10, 25, 50, 100, 250, 500]),
        'MLP_mean_squared_error': ('h5', 10, list(range(6))),
        'CNN_mean_squared_error_mpTrue_bnTrue': ('h5', 10, list(range(6)))
        }

print(classifiers)

if ('MLP_mean_squared_error' in classifiers.keys() or
        'CNN_mean_squared_error_mpTrue_bnTrue' in classifiers.keys()):
    print('Loading Keras')
    from keras.models import load_model

for classifier_name, (extension, n_runs, variables) in classifiers.items():
    # Load correct key and labels
    correct_byte = load_data.key_byte()
    (_, _, _, _, _, y_true) = load_data.load_data()
    test_length = len(y_true)

    # Load data
    if (classifier_name == 'RF' or
            classifier_name == 'MLP_mean_squared_error'):
        (_, _, _, _, x_test, _) = load_data.load_data_selected()
    elif classifier_name == 'CNN_mean_squared_error_mpTrue_bnTrue':
        (_, _, _, _, x_test, _) = load_data.load_data()
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Load key guesses mappings
    key_guesses = load_data.key_guesses()

    for current_variable in variables:
        # See how many classifiers we have (checking file existences)
        filepattern = (experiment_dir +
                       '{}_complexity{}_run{}.{}'.format(
                               classifier_name, current_variable,
                               '{i_bootstrap}', extension))

        filenames = PC.generatefilenames(
                filepattern, i_bootstrap=range(n_runs))

        filenames = sorted(filenames)

        count = 0
        valid_filenames = []
        for filename in filenames:
            if os.path.isfile(filename):
                count += 1
                valid_filenames.append(filename)

        if count >= 10:
            print('Found {} files for {}, argument {}.'.format(
                    count, classifier_name, current_variable))

            preds_filename = (experiment_dir +
                              '{}_complexity{}_preds.npy'.format(
                                      classifier_name, current_variable))
            GEs_filename = (experiment_dir +
                            '{}_complexity{}_GEs.npy'.format(
                                    classifier_name, current_variable))

            if (os.path.isfile(preds_filename) and
                    os.path.isfile(GEs_filename)):
                print('Predictions and GEs already computed, skipping')
                continue

            preds_y = np.zeros((count, test_length), dtype=int)
            GEs = np.zeros((count, test_length), dtype=float)

            for idx_file, valid_filename in enumerate(valid_filenames):
                if classifier_name == 'RF':
                    with open(valid_filename, 'rb') as f:
                        clf = pickle.load(f)
                        clf.n_jobs = 1

                        preds_y[idx_file, :] = clf.predict(x_test)
                        pred_ys = clf.predict_proba(x_test)

                        pred_ys = SCE.order_predict_probas(
                                pred_ys, key_guesses)
                        GEs[idx_file, :] = SCE.guessing_entropy(
                                pred_ys, correct_byte, n_experiments=200)

                if (classifier_name == 'CNN_mean_squared_error_mpTrue_bnTrue'
                        or classifier_name == 'MLP_mean_squared_error'):
                    clf = load_model(valid_filename)

                    preds_y[idx_file, :] = clf.predict_classes(x_test)
                    pred_ys = clf.predict_proba(x_test)

                    pred_ys = SCE.order_predict_probas(
                            pred_ys, key_guesses)
                    GEs[idx_file, :] = SCE.guessing_entropy(
                            pred_ys, correct_byte, n_experiments=200)

            # After making predictions and computing GEs, save results:
            print(preds_y)
            print(GEs)

            np.save(preds_filename, preds_y)
            np.save(GEs_filename, GEs)
            print('Saved for {} with complexity {}'.format(
                    classifier_name, current_variable))
