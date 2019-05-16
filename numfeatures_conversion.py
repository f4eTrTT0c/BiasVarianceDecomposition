# -*- coding: utf-8 -*-
"""
Get predictions and guessing entropies from saved classifiers
for a different number of selected features
"""
import os
import numpy as np
import pickle
import load_data
from sidechannel_tools import SideChannelEvaluator as SCE
from sidechannel_tools import PredictionsConverter as PC

# Directory for this experiment (read classifiers, write predictions/GEs)
experiment_dir = 'numfeatures/'

# classifier_name: (extension, n_runs, variables)
classifiers = {
        'RF100': ('pkl', 100),
        'MLP2_mean_squared_error': ('h5', 10)
        }

if 'MLP2_mean_squared_error' in classifiers.keys():
    print('Loading Keras')
    from keras.models import load_model

variables = [25, 50, 75, 100, 125, 150, 175, 200]

for classifier_name, (extension, n_runs) in classifiers.items():
    # Load correct key and labels
    correct_byte = load_data.key_byte()
    (_, _, _, _, x_test, y_true) = load_data.load_data_selected()
    test_length = len(y_true)

    # Load key guesses mappings
    key_guesses = load_data.key_guesses()

    for current_variable in variables:
        # See how many classifiers we have (checking file existences)
        filepattern = (experiment_dir +
                       '{}_{}features_run{}.{}'.format(
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
                              '{}_{}features_preds.npy'.format(
                                      classifier_name, current_variable))
            GEs_filename = (experiment_dir +
                            '{}_{}features_GEs.npy'.format(
                                    classifier_name, current_variable))

            if (os.path.isfile(preds_filename) and
                    os.path.isfile(GEs_filename)):
                print('Predictions and GEs already computed, skipping')
                continue

            preds_y = np.zeros((count, test_length), dtype=int)
            GEs = np.zeros((count, test_length), dtype=float)

            for idx_file, valid_filename in enumerate(valid_filenames):
                if classifier_name == 'RF100':
                    with open(valid_filename, 'rb') as f:
                        clf = pickle.load(f)
                        clf.n_jobs = 1

                        preds_y[idx_file, :] = clf.predict(
                                x_test[:, :current_variable])
                        pred_ys = clf.predict_proba(
                                x_test[:, :current_variable])

                        pred_ys = SCE.order_predict_probas(
                                pred_ys, key_guesses)
                        GEs[idx_file, :] = SCE.guessing_entropy(
                                pred_ys, correct_byte, n_experiments=200)
                if (classifier_name == 'MLP2_mean_squared_error'):
                    clf = load_model(valid_filename)

                    preds_y[idx_file, :] = clf.predict_classes(
                            x_test[:, :current_variable])

                    pred_ys = clf.predict_proba(
                            x_test[:, :current_variable])

                    pred_ys = SCE.order_predict_probas(
                            pred_ys, key_guesses)
                    GEs[idx_file, :] = SCE.guessing_entropy(
                            pred_ys, correct_byte, n_experiments=200)

            # After making predictions and computing GEs, save results:
            print(preds_y)
            print(GEs)

            np.save(preds_filename, preds_y)
            np.save(GEs_filename, GEs)
            print('Saved for {} with {} features'.format(
                    classifier_name, current_variable))
