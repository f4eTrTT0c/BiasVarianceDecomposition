# -*- coding: utf-8 -*-
"""
Plot Domingos and GE loss decomposition for classifiers
with different numbers of selected features
"""
import os
import numpy as np
import load_data
from bias_variance_computation import biasVarianceDomingos, GEVariance
from bias_variance_plots import plotBiasVarianceDomingos, plotGE3D

experiment_dir = 'numfeatures/'

# classifier_name: (extension, n_runs, variables)
classifiers = {
        'RF100': ('pkl', 100, 'RF100'),
        'MLP2_mean_squared_error': ('h5', 10, 'MLP2')
        }

variables = [25, 50, 75, 100, 125, 150, 175, 200]
x_caption = 'Features'
x_scale = 'linear'

for classifier_name, (extension, n_runs, clf_fancy) in classifiers.items():
    # Load correct labels
    (_, _, _, _, _, y_true) = load_data.load_data()
    test_length = len(y_true)

    domingos = {'L': [], 'B': [], 'V': [], 'Vu': [], 'Vb': [],
                'avg_accuracy': []}

    # First, plot Domingos' bias-variance decomposition
    for current_variable in variables:
        # Check if predictions can be found
        y_preds_filename = (experiment_dir +
                            '{}_{}features_preds.npy'.format(
                                    classifier_name, current_variable))

        if not os.path.isfile(y_preds_filename):
            print('Not found: {}'.format(y_preds_filename))
            break

        # Load predictions
        y_preds = np.load(y_preds_filename)
        results = biasVarianceDomingos(y_true, y_preds,
                                       include_avg_accuracy=True)
        for key, val in results.items():
            domingos[key].append(val)

    if len(domingos['L']) >= 2:
        domingos_file = (experiment_dir +
                         '{}_domingos.png'.format(clf_fancy))

        plotBiasVarianceDomingos(
                L=domingos['L'], B=domingos['B'], V=domingos['V'],
                Vu=domingos['Vu'], Vb=domingos['Vb'], x_label=x_caption,
                x_steps=variables[:len(domingos['L'])],
                avg_accuracy=domingos['avg_accuracy'], x_scale=x_scale,
                save_loc=domingos_file)

    # Second, plot 3D guessing entropy decomposition
    avgs = np.zeros((test_length, len(variables)))
    var_r = np.zeros((test_length, len(variables)))
    var_i = np.zeros((test_length, len(variables)))

    up_to_idx = 0
    for idx_variable, current_variable in enumerate(variables):
        GEs_filename = (experiment_dir +
                        '{}_{}features_GEs.npy'.format(
                                classifier_name, current_variable))
        if not os.path.isfile(GEs_filename):
            print('Not found: {}'.format(GEs_filename))
            break

        # Load predictions
        GEs = np.load(GEs_filename)
        avgs[:, idx_variable], var_r[:, idx_variable], \
            var_i[:, idx_variable] = GEVariance(GEs)

        up_to_idx = idx_variable

    if up_to_idx >= 2:
        GE3D_file = (experiment_dir + '{}_GE.png'.format(clf_fancy))

        # Only take up_to_idx scenarios
        plotGE3D(avgs[:, :up_to_idx],
                 var_r[:, :up_to_idx],
                 var_i[:, :up_to_idx],
                 x_caption, x_steps=variables[:up_to_idx],
                 save_loc=GE3D_file)
