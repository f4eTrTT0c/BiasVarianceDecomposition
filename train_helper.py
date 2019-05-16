# -*- coding: utf-8 -*-
"""
Helper functions for training and saving classifiers.
"""
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def save_model(location):
    """
    Function builder to save a classifier to a specified location.

    Parameters
    ----------
    location : str, the filename to write to. Should not include '.pkl'.
    When no directory is included, the file is written to the pwd.

    Returns
    -------
    save_model_def : function that uses a classifier to save predictions to
    a pre-determined file.

    Details:
        Parameters
        ----------
        clf : trained classifier

    """
    def save_model_def(clf):
        if not os.path.isfile(location + '.pkl'):
            loc = location + '.pkl'
        else:
            loc = location + '_r{}.pkl'.format(np.random.randint(10000))

        with open(loc, 'wb') as f:
            pickle.dump(clf, f)
            print('Saved {} to {}'.format(clf, f))

    return save_model_def


def RF_model_save_builder(location):
    """
    Function builder for saving trained Random Forests classifiers.

    Parameters
    ----------
    location : str, the filename to write to. Should not include '.npy'.
    When no directory is included, the file is written to the pwd.
    Should include at least dynamic parameter: '{i_bootstrap}' part,
    where the bootstrap number is filled in. An arbitrary number of named para-
    meters may be included (e.g. {num_trees}), which, should be filled in with
    the locpars parameters (see below) in the returned function.

    Returns
    -------
    RF_model_save : function which trains a RandomForestClassifier
    with the specified training data, ignoring the validation and test data,
    saving the classifier to a specified filename, filling in the location
    parameters.

    Function details:

        Parameters
        ----------
        i_bootstrap : int, number of the current bootstrap in the series
        x_train : training set (samples)
        y_train : training set (labels)
        x_validate : validation set (samples) -- ignored
        y_validate : validation set (labels) -- ignored
        x_test : test set (samples)
        y_test : test set (labels) -- ignored
        num_samples_save : number of samples in the test set to save results
        num_trees : int, number of trees in the forest
        locpars : dict, location parameters used to generate the filename

        Results
        -------
        Saves a numpy array with predict_probas of the test set.

        Returns
        -------
        Accuracy score : fraction of correctly classified samples over the
        entire test set.
    """
    def RF_model_save(i_bootstrap, x_train, y_train, x_validate,
                      y_validate, x_test, y_test, num_trees, locpars):
        model_saver = save_model(
                location.format(**{**locpars, 'i_bootstrap': i_bootstrap}))
        clf = RandomForestClassifier(num_trees)
        clf.fit(x_train, y_train)
        model_saver(clf)

        y_pred = clf.predict(x_test)
        return accuracy_score(y_test, y_pred)

    return RF_model_save
