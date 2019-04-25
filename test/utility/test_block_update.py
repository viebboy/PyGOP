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
from GOP.utility import gop_utils, gop_operators

INPUT_DIM = utility.INPUT_DIM
OUTPUT_DIM = utility.OUTPUT_DIM
BATCH_SIZE = 32
STEPS = 4


def test_block_update(tmpdir):
    model_path = os.path.join(tmpdir.dirname, 'test_model')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)

    params, model_data = utility.get_random_model_data()
    params['tmp_dir'] = tmpdir.dirname
    params['model_name'] = 'test_model'
    params['nodal_set'] = gop_operators.get_default_nodal_set()
    params['pool_set'] = gop_operators.get_default_pool_set()
    params['activation_set'] = gop_operators.get_default_activation_set()
    params['convergence_measure'] = random.choice(
        ['train_', 'val_']) + params['convergence_measure']
    block_names = ['gop_0_0', 'gop_1_0', 'bn_0_0', 'bn_1_0', 'output']

    all_op_sets = utility.get_all_operators()

    op_set_indices = {}
    for layer_name in model_data['op_sets'].keys():
        op_set_indices[layer_name] = all_op_sets.index(model_data['op_sets'][layer_name])

    train_states = {'topology': model_data['topology'],
                    'weights': model_data['weights'],
                    'op_set_indices': op_set_indices}

    _, _, new_weights = gop_utils.block_update_standalone(train_states,
                                                          params,
                                                          block_names,
                                                          utility.get_generator,
                                                          [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                          utility.get_generator,
                                                          [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                          utility.get_generator,
                                                          [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    for layer_name in new_weights.keys():
        if layer_name not in block_names:
            assert np.allclose(new_weights[layer_name][0], model_data['weights'][layer_name][0])

    shutil.rmtree(model_path)


if __name__ == '__main__':
    pytest.main([__file__])
