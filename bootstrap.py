# -*- coding: utf-8 -*-
"""
The Bootstrapper class offers a way to dynamically
evaluate classifiers using bootstrapping
"""
import numpy as np


class Bootstrapper:
    """Bootstrapper: generating bootstrapped datasets and executing a function.

    Bootstrapping is random resampling with replacement. It is useful to
    estimate statistical properties on classifiers, such as their accuracy
    score (mean and standard deviation). [1]

    Parameters
    ----------
    n_classes : the number of classes which should be represented in the data.

    x_train : training set (samples)

    y_train : training set (labels)

    x_validate : validation set (samples)

    y_validate : validation set (labels)

    x_test : test set (samples)

    y_test : test set (labels)

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

    """

    def __init__(self, n_classes, x_train, y_train, x_validate, y_validate,
                 x_test, y_test):
        self._x_train = x_train
        self._y_train = y_train
        self._x_validate = x_validate
        self._y_validate = y_validate
        self._x_test = x_test
        self._y_test = y_test
        self._n_classes = n_classes

        # Check if the number of classes is the same as those labels
        # presented in the training data: otherwise, the classifier will
        # 1) not learn very well, and 2) predict_proba in a wrong dimension
        check_classes = np.array_equal(
                np.array(range(0, self._n_classes)),
                np.unique(y_train))
        assert check_classes, \
            'Training set misses classes or expresses non-existing classes'

    def bootstrap_run(self, run_clf, n_bootstraps, n_samples=None,
                      n_features=None, **kwargs):
        """Generate bootstrapped sets and run the classifier procedure on them.

        Parameters
        ----------
        run_clf : function, handling the generated data. It should take the
        following positional arguments:
            i_bootstrap : number of the current bootstrap in the series
            x_train : training set (samples)
            y_train : training set (labels)
            x_validate : validation set (samples)
            y_validate : validation set (labels)
            x_test : test set (samples)
            y_test : test set (labels)
        Additionally, run_clf may have arbitrarely many other arguments, which
        are passed to it using **kwargs. The function may return anything.

        n_bootstraps : integer, number of bootstrapped sets to generate

        n_samples : integer or None. With None (default), generate training
        sets of the same size of the original set. With an integer value,
        this becomes the training set size.

        **kwargs : all of the parameters are passed to the run_clf features

        Returns
        -------
        outcomes : list of results from the classifier procedure, of length
        num_bootstraps (one item for each run).
        """
        outcomes = []

        if n_samples is None:
            n_samples = len(self._x_train)
        elif not isinstance(n_samples, int):
            raise ValueError('n_samples should either be None (use all data)'
                             ' or an integer specifying the training size')

        for i_bootstrap in range(n_bootstraps):
            # Take a bootstrapped sample
            need_resampling = True
            while need_resampling:
                bootstrap_indexes = np.random.choice(
                        range(len(self._x_train)),
                        size=n_samples,
                        replace=True)

                # if not all classes are present, try again
                need_resampling = not np.array_equal(
                        np.array(range(0, self._n_classes)),
                        np.unique(self._y_train[bootstrap_indexes]))

            # Run classifier procedure with the generated data
            outcomes.append(
                    run_clf(i_bootstrap,
                            self._x_train[bootstrap_indexes, :n_features],
                            self._y_train[bootstrap_indexes],
                            self._x_validate[:, :n_features],
                            self._y_validate,
                            self._x_test[:, :n_features],
                            self._y_test, **kwargs))

        return outcomes
