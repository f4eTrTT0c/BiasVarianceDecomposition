# -*- coding: utf-8 -*-
"""
Helper definitions to create Keras models for MLP and CNN networks.
"""

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D


def flexible_MLP(num_features, num_classes, num_hidden_layers):
    """
    Create an MLP based on the number of features, classes and hidden layers.

    Parameters
    ----------
    num_features : int, number of features as input layer.

    num_classes : int, number of neurons in output layer.

    num_hidden_layers : int, number of required hidden layers; each hidden
    will have 200 neurons with ReLu activation.

    Returns
    -------
    model : Keras untrained and uncompiled MLP model with the requested
    architecture:
        - An input layer with num_features inputs.
        - Hidden layer with 200 neurons with ReLu activation, repeated
          num_hidden_layers times
        - An output layer of num_classes output neurons with softmax
          activation.
    """
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(num_features,)))

    for i in range(num_hidden_layers):
        model.add(Dense(200, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    return model


def flexible_CNN_add_block(model, block_no, batch_normalization_in_block,
                           max_pooling_in_block):
    if block_no == 0:
        return
    elif block_no == 1:
        model.add(Conv1D(
                filters=8, kernel_size=3, strides=1, activation='relu'))
    elif block_no == 2:
        model.add(Conv1D(
                filters=16, kernel_size=3, strides=1, activation='relu'))
    elif block_no == 3:
        model.add(Conv1D(
                filters=32, kernel_size=3, strides=1, activation='relu'))
    elif block_no == 4:
        model.add(Conv1D(
                filters=64, kernel_size=3, strides=1, activation='relu'))
        model.add(Conv1D(
                filters=64, kernel_size=3, strides=1, activation='relu'))
    elif block_no == 5:
        model.add(Conv1D(
                filters=128, kernel_size=3, strides=1, activation='relu'))
        model.add(Conv1D(
                filters=128, kernel_size=3, strides=1, activation='relu'))
    elif block_no == 6:
        model.add(Conv1D(
                filters=256, kernel_size=3, strides=1, activation='relu'))
        model.add(Conv1D(
                filters=256, kernel_size=3, strides=1, activation='relu'))
    else:
        raise ValueError("Too high convolutional block nr" +
                         " ({}), should not exceed 6.".format(block_no))

    # Add the end of the convolutional block, add batch normalization
    # and max pooling if desired
    if batch_normalization_in_block:
        model.add(BatchNormalization())

    if max_pooling_in_block:
        model.add(MaxPooling1D(pool_size=2, strides=2))


def flexible_CNN(num_features, num_classes, num_convolutional_blocks,
                 batch_normalization_in_block=True, max_pooling_in_block=True):
    """
    Create a CNN based on the number of features, classes and hidden layers.

    Parameters
    ----------
    num_features : int, number of features as input layer.

    num_classes : int, number of neurons in output layer.

    num_convolutional_blocks : int, number of required convolutional block.

    batch_normalization_in_block : boolean, indicating whether each
    convolutional block should have batch normalization (reduces overfitting).

    max_pooling_in_block : boolean, indicating whether each convolutional block
    should have a max pooling (reduces model complexity).

    Returns
    -------
    model : Keras untrained and uncompiled model with the requested
    architecture:
        - An input layer with num_features inputs.
        - Batch Normalization layer.
        - 0 .. 6 (depending on num_convolutional_blocks) convolutional blocks
          with a convolutional layer and, if requested, batch normalization and
          max pooling. Convolutional layers have ReLu activation.
        - A dropout layer (50%) using the flattend convoluted data.
        - A dense layer of 512 neurons with ReLu activation.
        - A dropout layer (50%).
        - An output layer of num_classes output neurons with softmax
          activation

    """
    model = Sequential()

    # Batch normalization before feeding the data to the first layer
    model.add(BatchNormalization(input_shape=(num_features, 1)))

    # For each convolutional block, add a convolutional layer
    # and, depending on parameters, include batch normalization and max pooling
    # to avoid overfitting and reduce the model complexity
    for i in range(1, num_convolutional_blocks + 1):
        flexible_CNN_add_block(
                model, i, batch_normalization_in_block, max_pooling_in_block)

    # Flatten the output of the last convolutional block
    model.add(Flatten())

    # Final hidden layer: fully connected, avoiding overfitting using dropout
    model.add(Dropout(rate=0.5))
    model.add(Dense(512, activation='relu'))

    model.add(Dropout(rate=0.5))

    # Output layer, normalized to class probabilities using softmax
    model.add(Dense(num_classes, activation='softmax'))

    return model
