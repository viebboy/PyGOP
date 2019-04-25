# -*- coding: utf-8 -*-
"""
Utility for testing purpose


Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import numpy as np
import random
from keras import backend as K
from GOP.utility import gop_operators


def get_generator(data):

    input_dim, output_dim, batch_size, steps, seed = data

    if seed is not None:
        np.random.seed(seed)

    x = np.random.rand(batch_size * steps, input_dim)
    y = np.random.rand(batch_size * steps, output_dim) if output_dim is not None else None

    def gen():
        while True:
            indices = list(range(batch_size * steps))
            random.shuffle(indices)

            for i in range(steps):
                start_idx = i * batch_size
                stop_idx = (i + 1) * batch_size
                if output_dim is not None:
                    yield x[indices[start_idx:stop_idx]], y[indices[start_idx:stop_idx]]
                else:
                    yield x[indices[start_idx:stop_idx]]

    return gen(), steps


def get_test_generator(data):
    x, y, batch_size = data
    N = x.shape[0]
    steps = int(np.ceil(N / float(batch_size)))

    def gen():
        while True:
            for i in range(steps):
                start_idx = i * batch_size
                stop_idx = min(N, (i + 1) * batch_size)
                if y is not None:
                    yield x[start_idx:stop_idx], y[start_idx:stop_idx]
                else:
                    yield x[start_idx:stop_idx]

    return gen(), steps


def mean_absolute_error_numpy(y_true, y_pred):
    return np.mean(np.abs(y_true.flatten() - y_pred.flatten()))


def mean_absolute_error_keras(y_true, y_pred):
    return K.mean(K.abs(K.flatten(y_true) - K.flatten(y_pred)))


def custom_nodal(x, w):
    return K.exp(-K.square(x * w))


def custom_pool(z):
    return K.min(z, axis=1)


def custom_activation(y):
    return K.relu(y)


def get_nodal_set():
    nodal_set = gop_operators.get_default_nodal_set()[:1] + [custom_nodal]
    return nodal_set


def get_pool_set():
    pool_set = gop_operators.get_default_pool_set()[:1] + [custom_pool]
    return pool_set


def get_activation_set():
    activation_set = [custom_activation]
    return activation_set
