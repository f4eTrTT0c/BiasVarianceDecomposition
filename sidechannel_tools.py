# -*- coding: utf-8 -*-
"""
Classes SideChannelEvaluator (compute (partial) guessing entropy) and
PredictionsConverter (read out files)
"""
import numpy as np
from sklearn.model_selection import ParameterGrid


class SideChannelEvaluator:
    """
    Class to help compute guessing entropy, success rate, etc.
    """
    def partial_guessing_entropy(ordered_key_probas_sums,
                                 correct_key_byte):
        """
        Computes partial guessing entropy.

        Parameters
        ----------
        ordered_key_probas_sums : array of 256 values, indicating the sum of
        predicted probabilities for each key guess

        correct_key_byte : int, indicating the correct key byte (e.g. index of
        ordered_key_probas_sums relating to the real key)
        """
        ordered_guesses = np.argsort(ordered_key_probas_sums)[::-1]
        key_rank = np.where(ordered_guesses == correct_key_byte)[0][0]
        return key_rank

    def guessing_entropy(ordered_key_probas, correct_key_byte,
                         n_experiments=100):
        """
        Computes guessing entropy.

        Parameters
        ----------
        ordered_key_probas : n x 256 nparray: the assigned probabily per trace,
        classes ordered for each key guess.

        correct_key_byte : int, correct key byte (e.g. column number of
        ordered_key_probas indicating the correct class)

        n_experiments : number of runs; this means the number of permutations
        to the indexes (default 100). The function returns the average result
        of these experiments. With 1 run, the function evaluates partial
        guessing entropy.

        Returns
        -------
        Guessing entropy : 1 x n nparray, indicating the rank of the correct
        key for each #traces, starting with 1 trace and ending with n traces
        """

        n_traces = len(ordered_key_probas)

        # Partial guessing entropies
        PGEs = np.zeros((n_experiments, n_traces))
        indexes = np.arange(n_traces)

        for experiment_no in range(n_experiments):
            # Randomly shuffle indexes to create a new experiment
            np.random.shuffle(indexes)

            key_probas_sum = np.zeros(256)

            for trace_no in range(n_traces):
                key_probas_sum += ordered_key_probas[indexes[trace_no]]
                PGEs[experiment_no, trace_no] = \
                    SideChannelEvaluator.partial_guessing_entropy(
                            key_probas_sum, correct_key_byte)

        GEs = PGEs.mean(axis=0)

        return GEs

    def order_predict_probas_value(predict_probas, guesses_to_labels):
        """
        From an array of predicted class probabilities (intermediate values),
        to an array of the same size with predicted probabilities per key guess

        Parameters
        ----------
        predict_probas : n x 256 array of class probabilities (interm. values)

        guesses_to_labels : n x 256 array, mapping the intermediate values
        to key guesses. Example: if the key byte is 24, one-dimensional array
        labels_to_key_guesses[:, 24] would relate to the correct class labels
        for each trace.
        """
        guesses_to_labels = guesses_to_labels.astype(int)
        assert predict_probas.shape[1] == 256, "Should represent 256 classes!"
        assert predict_probas.shape == guesses_to_labels.shape, \
            "Parameters' shapes should be equal, but found {} and {}".format(
                    predict_probas.shape, guesses_to_labels.shape)

        pred_ordered = np.zeros(predict_probas.shape)
        for guess in range(256):
            pred_ordered[:, guess] = [values[index] for values, index in zip(
                    predict_probas, guesses_to_labels[:, guess])]

        return pred_ordered

    def order_predict_probas_HW(predict_probas, guesses_to_labels):
        """
        From an array of predicted class probabilities (HW), to an array of the
        same length x 256 with predicted probabilities per key guess

        Parameters
        ----------
        predict_probas : n x 9 array of class probabilities (HWs)

        guesses_to_labels : n x 256 array, mapping the intermediate values
        to key guesses. Example: if the key byte is 24, one-dimensional array
        labels_to_key_guesses[:, 24] would relate to the correct class labels
        for each trace.
        """
        guesses_to_labels = guesses_to_labels.astype(int)
        assert predict_probas.shape[1] == 9, "Should represent 9 classes!"
        assert predict_probas.shape[0] == guesses_to_labels.shape[0], \
            "Parameters' should have equal length, but found {} and {}".format(
                    predict_probas.shape[0], guesses_to_labels.shape[0])

        pred_ordered = np.zeros(guesses_to_labels.shape)
        for guess in range(256):
            pred_ordered[:, guess] = [values[index] for values, index in zip(
                    predict_probas, guesses_to_labels[:, guess])]

        return pred_ordered

    def order_predict_probas(predict_probas, guesses_to_labels):
        """
        From an array of predicted class probabilities (HW/value), to an array
        of the same length x 256 with predicted probabilities per key guess

        Parameters
        ----------
        predict_probas : n x 9 array of class probabilities (HWs), or
        n x 256 array of class probabilities (interm. values)

        guesses_to_labels : n x 256 array, mapping the intermediate values
        to key guesses. Example: if the key byte is 24, one-dimensional array
        labels_to_key_guesses[:, 24] would relate to the correct class labels
        for each trace.
        """
        if predict_probas.shape[1] == 9:
            return SideChannelEvaluator.order_predict_probas_HW(
                    predict_probas, guesses_to_labels)
        elif predict_probas.shape[1] == 256:
            return SideChannelEvaluator.order_predict_probas_value(
                    predict_probas, guesses_to_labels)
        else:
            raise ValueError('predict_probas should either have '
                             '9 or 256 columns, but found {}'.format(
                                     predict_probas.shape[1]))


class PredictionsConverter:
    """Class that helps to convert predictions from bias variance scripts."""

    def predict_proba_files_to_key_rank_array(y_true, n_classes,
                                              predict_proba_filenames,
                                              save_file=None):
        """
        Conversion from file(s) of predictions to single nparray with the
        rank of the correct label.

        Parameters
        ----------
        y_true : 1-dimensional (np)array, true labels of which the ranking
        within the predictions should be determined.
        n_classes : int, typically 9 (HW) or 256 (value)
        predict_proba_filenames : iterable with filenames. Filename may include
        directory; if missing, the script looks in the pwd. Files should
        contain a numpy array of size len(y_true) * n_classes.
        new_filename : str or None (default). When None is specified, the
        resulting nparray is not saved, only returned. When new_filename is a
        string, the result is saved to this filename.
        """
        n_classifiers = len(predict_proba_filenames)
        n_samples = len(y_true)
        key_ranks = np.full((n_classifiers, n_samples), n_classes)

        for classifier_no, classifier_filename in enumerate(
                predict_proba_filenames):
            # Load file
            pred_proba = np.load(classifier_filename)

            # Sanity check: should be of shape len(y_true) * n_classes
            assert pred_proba.shape == (n_samples, n_classes), \
                "pred_proba shape should be ({}, {}), but is {}".format(
                        n_samples, n_classes, pred_proba.shape)

            # Run argsort to find which
            correct_positions = np.flip(np.argsort(pred_proba, axis=1), axis=1)
            key_ranks[classifier_no, :] = correct_positions[
                    np.arange(len(correct_positions)), y_true]

        if save_file is not None:
            np.save(save_file, key_ranks)

        return key_ranks

    def generatefilenames(filename_pattern, **kwargs):
        """
        From a specified pattern and a trivial number of arguments,
        generate all filenames in the grid for all combinations of arguments.

        Parameters
        ----------
        filename_pattern : str, including parameters
        (e.g., '/dir/results{model}.npy').

        kwargs : list of values to fill in, for each parameter
        in the filename_pattern.

        Returns
        -------
        A list of filenames, with all combinations of parameters included.

        Example
        -------
        >> pattern = '{model}_{keep_blank}_{dataset_name}_run{run}.npy'
        >> options = {
               'model': ['HW', 'value'],
               'dataset_name': ['DPAv4'],
               'run': range(3),
               'keep_blank': ['{keep_blank}']}
        >> print(predictionsConverter.generatefilenames(pattern, **options))
            ['HW_{keep_blank}_DPAv4_run0.npy',
             'HW_{keep_blank}_DPAv4_run1.npy',
             'HW_{keep_blank}_DPAv4_run2.npy',
             'value_{keep_blank}_DPAv4_run0.npy',
             'value_{keep_blank}_DPAv4_run1.npy',
             'value_{keep_blank}_DPAv4_run2.npy']
        -------

        """
        grid = list(ParameterGrid(kwargs))
        return [filename_pattern.format(**params) for params in grid]

    def order_predict_probas_value(predict_probas, guesses_to_labels):
        """
        From an array of predicted class probabilities (intermediate values),
        to an array of the same size with predicted probabilities per key guess

        Parameters
        ----------
        predict_probas : n x 256 array of class probabilities (interm. values)

        labels_to_key_guesses : n x 256 array, mapping the intermediate values
        to key guesses. Example: if the key byte is 24, one-dimensional array
        labels_to_key_guesses[:, 24] would relate to the correct class labels
        for each trace.
        """
        guesses_to_labels = guesses_to_labels.astype(int)
        assert predict_probas.shape[1] == 256, "Should represent 256 classes!"
        assert predict_probas.shape == guesses_to_labels.shape, \
            "Parameters' shapes should be equal, but found {} and {}".format(
                    predict_probas.shape, guesses_to_labels.shape)

        pred_ordered = np.zeros(predict_probas.shape)
        for guess in range(256):
            pred_ordered[:, guess] = [values[index] for values, index in zip(
                    predict_probas, guesses_to_labels[:, guess])]

        return pred_ordered

    def predictions_to_GEs(filenames, guesses_to_labels, correct_key_byte,
                           n_experiments=100):
        """
        Given an iterable of file names, load predict_probas from each file and
        compute the guessing entropy.

        Parameters
        ----------
        filenames : iterable, list of file names to load predictions from. Each
        file should contain an n x 256 array of class probabilities
        (intermediate values)

        guesses_to_labels : n x 256 array, mapping the intermediate values
        to key guesses. Example: if the key byte is 24, one-dimensional array
        labels_to_key_guesses[:, 24] would relate to the correct class labels
        for each trace.

        n_experiments : int, number of experiments to compute the GE with
        (default 100)

        Returns
        -------
        len(filenames) x n nparray, containing the guessing entropy for each
        file for 1 .. n traces in the attack set

        """
        n_runs = len(filenames)
        GEs = np.zeros((n_runs, len(guesses_to_labels)))

        for run_i, filename in enumerate(filenames):
            pred = np.load(filename)
            pred_ordered = SideChannelEvaluator.order_predict_probas_value(
                    pred, guesses_to_labels)
            GEs[run_i, :] = SideChannelEvaluator.guessing_entropy(
                    pred_ordered, correct_key_byte, n_experiments)

        return GEs
