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
from GOP.utility import gop_utils

INPUT_DIM = utility.INPUT_DIM
OUTPUT_DIM = utility.OUTPUT_DIM
BATCH_SIZE = 32
STEPS = 4


def test_calculate_memory(tmpdir):
    model_path = os.path.join(tmpdir.dirname, 'test_model')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)

    params, train_states = utility.get_random_popmem_states()
    params['tmp_dir'] = tmpdir.dirname
    params['model_name'] = 'test_model'
    params['search_computation'] = ('cpu', 1)

    params['memory_type'] = 'PCA'
    gop_utils.calculate_memory_block_standalone(params,
                                                train_states,
                                                utility.get_generator,
                                                [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                utility.get_generator,
                                                [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                utility.get_generator,
                                                [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    shutil.rmtree(model_path)
    os.mkdir(model_path)

    params['memory_type'] = 'LDA'
    gop_utils.calculate_memory_block_standalone(params,
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
