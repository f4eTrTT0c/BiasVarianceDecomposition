# -*- coding: utf-8 -*-
"""
Compute Domingos and GE decomposition.
"""
import numpy as np


def biasVarianceDomingos(y_true, y_preds, include_avg_accuracy=True):
    """
    Compute Domingos bias-variance decomposition for 0-1 loss. See [1].

    Parameters
    ----------
    y_true : numpy array of length n, true labels
    y_preds: numpy array of size l x n, predicted labels by the l
    classifiers

    Returns
    -------
    result : dict with keys 'L', 'B', 'V', 'Vu', 'Vb' indicating total
    loss, bias, total variance, unbiased and biased variance, respectively
    Also includes 'avg_accuracy' if requested.

    References
    ----------
    [1] Domingos,P.:A unified bias-variance decomposition and its
    applications. In: Langley, P. (ed.) Proceedings of the Seventeenth
    International Conference on Machine Learning (ICML 2000),
    Stanford University, Stanford, CA, USA, June 29–July 2, 2000,
    pp. 231–238. Morgan Kaufmann (2000)
    """

    assert len(y_true.shape) == 1, 'y_true should be 1-dimensional'
    assert len(y_true) == y_preds.shape[1], \
        'y_true and y_preds have different test set sizes: {} and {}'.format(
                len(y_true), y_preds.shape[1])

    y_true, y_preds = y_true.astype(int), y_preds.astype(int)

    test_length = len(y_true)
    num_bootstraps = y_preds.shape[0]

    mean_predictions = np.zeros(shape=(test_length), dtype=int)
    bias_point = np.zeros(shape=(test_length), dtype=int)
    var_point = np.zeros(shape=(test_length), dtype=float)
    c2_point = np.ones(shape=(test_length), dtype=float)
    var_point_to_ratio = np.zeros(shape=(test_length), dtype=float)

    for i in range(test_length):
        # For each point, compute the mean prediction, bias and variance
        mean_predictions[i] = np.argmax(np.bincount(y_preds[:, i]))
        bias_point[i] = 0 if mean_predictions[i] == y_true[i] else 1
        var_point[i] = np.sum(
                [0 if x == mean_predictions[i] else 1 for x in y_preds[:, i]]
                )/num_bootstraps

        # For this point, compute c2: 1 when unbiased, otherwise:
        if bias_point[i] == 1:
            # If the bias is one, the mean prediction is wrong:
            #   of the outputted labels that are different from the mean
            #   prediction, compute the fraction of correct answers.
            # This fraction, negative, is the c2 coefficient: this fraction of
            #   the variance will decrease the error.
            c2_val = 0
            c2_count = 0
            for pred in range(num_bootstraps):
                if y_preds[pred, i] != mean_predictions[i]:
                    c2_count += 1
                    if y_preds[pred, i] == y_true[i]:
                        c2_val += 1
            if c2_count != 0:
                c2_val /= c2_count

            c2_point[i] = c2_val

    var_point_to_ratio = var_point * c2_point

    u_var_mean = np.sum(var_point_to_ratio[bias_point == 0])/test_length
    b_var_mean = np.sum(var_point_to_ratio[bias_point == 1])/test_length
    var_mean = u_var_mean - b_var_mean
    bias_mean = np.mean(bias_point)

    loss = var_mean + bias_mean

    # Save to arrays
    result = {
            'L': loss,
            'B': bias_mean,
            'V': var_mean,
            'Vu': u_var_mean,
            'Vb': b_var_mean
            }

    if include_avg_accuracy:
        # Compute classical accuracy for each classifier
        accuracy_scores = np.sum(y_preds == y_true, axis=1) / test_length
        # Output average
        result['avg_accuracy'] = np.mean(accuracy_scores)

    return result


def GEVariance(GEs):
    """
    Compute GE bias-variance decomposition as proposed in our 'Bias-variance
    Decomposition in Machine Learning-based Side-channel analysis' paper.

    Parameters
    ----------
    GEs : k x M numpy array, where GEs[i, j] indicates the Guessing Entropy for
    run i and scenario (e.g., training set size) number j

    Returns
    -------
    avgs : numpy array of length M, indicating mean GE (bias)
    var_r : numpy array of length M, indicating the plane of loss-reducing
    variance
    var_i : numpy array of length M, indicating the plane of loss-inducing
    variance
    """
    avgs = GEs.mean(axis=0)
    stds = GEs.std(axis=0)

    var_r = avgs - stds
    var_i = avgs + stds
    return (avgs, var_r, var_i)
