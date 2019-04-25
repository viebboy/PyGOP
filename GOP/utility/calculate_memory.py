#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Runable to perform block update

Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""
from __future__ import print_function

import pickle
import dill
from GOP.utility import gop_utils, misc
import sys
import os

module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if module_path not in sys.path:
    sys.path.append(module_path)


def main(argv):

    path = os.environ['memory_block_path']

    with open(os.path.join(path, 'params.pickle'), 'rb') as f:
        params = dill.load(f)

    path = os.path.join(params['tmp_dir'], params['model_name'])
    train_func, train_data, val_func, val_data, test_func, test_data = misc.unpickle_generator(
        os.path.join(path, 'data.pickle'))

    params = misc.reconstruct_parameters(params)

    with open(os.path.join(path, 'train_states_tmp.pickle'), 'rb') as f:
        train_states = pickle.load(f)

    pre_bn_weight, projection, post_bn_weight = gop_utils.calculate_memory_block(params,
                                                                                 train_states,
                                                                                 train_func,
                                                                                 train_data)

    """ write results """

    result = {'pre_bn_weight': pre_bn_weight,
              'projection': projection,
              'post_bn_weight': post_bn_weight}

    with open(os.path.join(path, 'memory_block.pickle'), 'wb') as fid:
        pickle.dump(result, fid)

    with open(os.path.join(path, 'memory_block_finish.txt'), 'a') as fid:
        fid.write('x')


if __name__ == "__main__":
    main(sys.argv[1:])
