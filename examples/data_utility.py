#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import os
import sys
import random
import numpy as np

# check and add current path
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if cwd not in sys.path:
    sys.path.append(cwd)


def load_miniCelebA(arguments):
    """
    Data loading function of miniCelebA to be used with PyGOP's algorithms

    Args:
        arguments (list): A list of arguments including:
                            - x_file (string): path to X (.npy file)
                            - y_file (string): path to Y (.npy file)
                            - batch_size (int): size of mini batch
                            - shuffle (bool): whether to shuffle minibatches

    Returns:
        gen (generator): python generator that generates mini batches of (x,y)
        steps (int): number of mini batches in the whole data

    """

    x_file, y_file, batch_size, shuffle = arguments
    X = np.load(x_file)
    Y = np.load(y_file)

    N = X.shape[0]
    steps = int(np.ceil(float(N) / batch_size))

    def gen():
        indices = list(range(N))
        while True:
            if shuffle:
                random.shuffle(indices)

            for step in range(steps):
                start_idx = step * batch_size
                stop_idx = min(N, (step + 1) * batch_size)
                batch_indices = indices[start_idx:stop_idx]

                yield X[batch_indices], Y[batch_indices]

    return gen(), steps
