# -*- coding: utf-8 -*-
"""
Helper functions to train and save neural networks, given certain training data
and parameters.
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras_architectures import flexible_MLP, flexible_CNN
from keras.optimizers import Adam

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None,
            decay=0.0, amsgrad=False)


def save_model_DL(location):
    """
    Function builder to save a Keras model to a specified location.

    Parameters
    ----------
    location : str, the filename to write to. Should not include '.h5'.
    When no directory is included, the file is written to the pwd.

    Returns
    -------
    save_model_def : function that uses a classifier to save predictions to
    a pre-determined file.

    Details:
        Parameters
        ----------
        clf : trained Keras model

    """
    def save_model_def(clf):
        if not os.path.isfile(location + '.h5'):
            loc = location + '.h5'
        else:
            loc = location + '_r{}.h5'.format(np.random.randint(10000))

        clf.save(loc)

    return save_model_def


def MLP_model_save_builder(location):
    """Function builder for saving MLPs.

    Parameters
    ----------
    location : str, the filename to write to. Should not include '.h5'.
    When no directory is included, the file is written to the pwd.
    Should include at least dynamic parameter: '{i_bootstrap}' part,
    where the bootstrap number is filled in. An arbitrary number of named para-
    meters may be included (e.g. {num_layers}), which, should be filled in with
    the locpars parameters (see below) in the returned function.

    Returns
    -------
    MLP_predict_proba_save : function which trains an MLP
    with the specified training data, ignoring the validation data, saving the
    model to a specified filename, filling in the location parameters,
    and returns accuracy score over the entire test set.

    Function details:

        Parameters
        ----------
        i_bootstrap : int, number of the current bootstrap in the series
        x_train : training set (samples)
        y_train : training set (labels)
        x_validate : validation set (samples)
        y_validate : validation set (labels)
        x_test : test set (samples)
        y_test : test set (labels) -- ignored
        num_samples_save : number of samples in the test set to save results

        num_classes : int, number of classes (model output shape)
        num_hidden_layers : int, number of hidden layers
        loss_function : str, any valid Keras loss function
            (e.g. 'categorical_crossentropy', 'mean_squared_error')
        num_epochs : number of epochs
        batch_size : batch size (should be corrected for #features)
        locpars : dict, location parameters used to generate the filename

        Results
        -------
        Saves a numpy array with predict_probas of the test set.

        Returns
        -------
        Accuracy score : fraction of correctly classified samples over the
        entire test set.
    """
    def MLP_model_save(i_bootstrap, x_train, y_train, x_validate, y_validate,
                       x_test, y_test, num_classes, num_hidden_layers,
                       loss_function, num_epochs, batch_size, locpars):
        model_saver = save_model_DL(
                location.format(**{**locpars, 'i_bootstrap': i_bootstrap}))

        num_features = x_train.shape[1]

        # Hot-encoding class labels
        y_train_cats = to_categorical(y_train, num_classes)
        y_validate_cats = to_categorical(y_validate, num_classes)

        # Create, compile, show and fit model
        clf = flexible_MLP(num_features, num_classes, num_hidden_layers)
        clf.compile(optimizer=adam, loss=loss_function,
                    metrics=['accuracy'])
        clf.summary()
        clf.fit(x_train, y_train_cats,
                validation_data=(x_validate, y_validate_cats),
                epochs=num_epochs, batch_size=batch_size, verbose=2)

        # Save predictions
        model_saver(clf)

        y_pred = clf.predict_classes(x_test)
        return accuracy_score(y_test, y_pred)

    return MLP_model_save


def CNN_model_save_builder(location):
    """Function builder for saving CNNs.

    Read more
    ---------
    See help(MLP_model_save_builder)
    """
    def CNN_model_save(i_bootstrap, x_train, y_train, x_validate, y_validate,
                       x_test, y_test, num_classes, num_convblocks,
                       loss_function, num_epochs, batch_size, BN_in_block,
                       max_pooling_in_block, locpars):
        model_saver = save_model_DL(
                location.format(**{**locpars, 'i_bootstrap': i_bootstrap}))

        num_features = x_train.shape[1]

        # Hot-encoding class labels
        y_train_cats = to_categorical(y_train, num_classes)
        y_validate_cats = to_categorical(y_validate, num_classes)

        # Create, compile, show and fit model
        clf = flexible_CNN(num_features, num_classes, num_convblocks,
                           BN_in_block, max_pooling_in_block)
        clf.compile(optimizer=adam, loss=loss_function,
                    metrics=['accuracy'])
        clf.summary()
        clf.fit(x_train, y_train_cats,
                validation_data=(x_validate, y_validate_cats),
                epochs=num_epochs, batch_size=batch_size, verbose=2)

        # Save predictions
        model_saver(clf)

        y_pred = clf.predict_classes(x_test)
        return accuracy_score(y_test, y_pred)

    return CNN_model_save
