# -*- coding: utf-8 -*-
"""
Functions to load data. All functions must be implemented to use the scripts
in this repository.
"""
import numpy as np


def load_data():
    """
    Load training, validation and test set to numpy arrays.

    Features are expected to be in chronologic order, so that the CNN can apply
    its convolution to the data.

    Usage
    -----
    (x_train, y_train, x_validate, y_validate, x_test, y_test) = load_data()

    Returns
    -------
    x_train : N x p numpy array if the number of training traces is N, which
    each tracing having p features. Training features.

    y_train : numpy array of N integers, denotes the profiling labels, based on
    any leakage model (e.g. HW or intermediate value). Training labels.

    x_validate, y_validate, x_test, y_test: analogous to x_train and y_train,
    for validation and testing (attack) traces
    """
    raise NotImplementedError('Please implement functions in load_data.py')

    return (x_train, y_train, x_validate, y_validate, x_test, y_test)


def load_data_selected():
    """
    Load training, validation and test set to numpy arrays, where features are
    ordered according to some rank, with the best feature first. In our paper,
    we used the absolute value of the Pearson correlation to select 200
    features.

    Usage
    -----
    (x_train, y_train, x_validate, y_validate, x_test, y_test) =
        load_data_selected()

    Returns
    -------
    x_train : N x p numpy array if the number of training traces is N, which
    each tracing having p features. Features are sorted according to some
    metric.

    y_train : numpy array of N integers, denotes the profiling labels, based on
    any leakage model (e.g. HW or intermediate value). Training labels.

    x_validate, y_validate, x_test, y_test: analogous to x_train and y_train,
    for validation and testing (attack) traces -- same order of features as
    x_train.
    """
    raise NotImplementedError('Please implement functions in load_data.py')

    return (x_train, y_train, x_validate, y_validate, x_test, y_test)


def key_guesses():
    """
    Load a mapping for each key guess, for all traces, to a leakage model
    label. Returns a numpy array of integers of size M x 256, where M is the
    size of the test set. The columns in this array indicate key candidates
    (i.e., column 0 corresponds to key byte 0). The values in the array should
    correspond with the leakage label for each trace in the test set.

    Value model: key_guess[i, j] = Sbox(Pi ^ j)
    HW model:    key_guess[i, j] = HW(Sbox(Pi ^ j))
        with Pi indicating the plaintext byte.

    This array is an integer array.

    Returns
    -------
    guesses : numpy array of integers of size M x 256
    """
    raise NotImplementedError('Please implement functions in load_data.py')
    return guesses


def key_byte():
    """
    Get the correct key byte of the test (attack) set.

    Returns
    -------
    byte : integer in range [0, 255], correct key byte
    """
    raise NotImplementedError('Please implement functions in load_data.py')
    return byte


def n_classes():
    """
    Get the number of classes in the leakage model

    Returns
    -------
    byte : integer, number of classes in the leakage model, typically 9 (HW) or
    256 (value)
    """
    raise NotImplementedError('Please implement functions in load_data.py')
    return num_classes
