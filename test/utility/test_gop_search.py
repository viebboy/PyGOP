#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import pytest
import os
import shutil
import utility
import numpy as np
import random
from GOP.utility import gop_utils

INPUT_DIM = utility.INPUT_DIM
OUTPUT_DIM = utility.OUTPUT_DIM
BATCH_SIZE = 32
STEPS = 4


def test_search_cpu(tmpdir):
    model_path = os.path.join(tmpdir.dirname, 'test_model')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)

    # test HeMLGOP, HoMLGOP, HeMLRN, HoMLRN
    params, train_states = utility.get_random_states()

    params['tmp_dir'] = tmpdir.dirname
    params['model_name'] = 'test_model'
    params['convergence_measure'] = random.choice(
        ['train_', 'val_']) + params['convergence_measure']
    params['no_op_set'] = len(params['nodal_set']) * \
        len(params['pool_set']) * len(params['activation_set'])
    params['search_computation'] = ('cpu', 2)
    train_states['model'] = random.choice(['HeMLGOP', 'HoMLGOP', 'HeMLRN', 'HoMLRN'])

    gop_utils.search_cpu(params,
                         train_states,
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    shutil.rmtree(model_path)

    # test POPfast
    params['max_topology'] = [20, 20, 20]
    params['layer_iter'] = 1
    params['block_iter'] = 0
    train_states['model'] = 'POPfast'

    os.mkdir(model_path)
    gop_utils.search_cpu(params,
                         train_states,
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    shutil.rmtree(model_path)

    # test POP
    train_states['model'] = 'POP'
    train_states['topology'][-1] = ('gop', OUTPUT_DIM)
    os.mkdir(model_path)

    # with GIS for output layer
    train_states['search_layer'] = 'output'
    train_states['hidden_op_set_idx'] = np.random.randint(0, params['no_op_set'])

    gop_utils.search_cpu(params,
                         train_states,
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    # with GIS for hidden layer
    train_states['search_layer'] = 'hidden'
    train_states['output_op_set_idx'] = np.random.randint(0, params['no_op_set'])

    gop_utils.search_cpu(params,
                         train_states,
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    shutil.rmtree(model_path)


if __name__ == '__main__':
    pytest.main([__file__])
