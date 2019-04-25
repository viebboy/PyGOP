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

    path = os.environ['block_update_path']

    with open(os.path.join(path, 'params.pickle'), 'rb') as f:
        params = dill.load(f)

    path = os.path.join(params['tmp_dir'], params['model_name'])
    train_func, train_data, val_func, val_data, test_func, test_data = misc.unpickle_generator(
        os.path.join(path, 'data.pickle'))

    params = misc.reconstruct_parameters(params)

    with open(os.path.join(path, 'train_states_tmp.pickle'), 'rb') as f:
        train_states = pickle.load(f)

    block_names = train_states['block_names']

    measure, history, weights = gop_utils.block_update(train_states['topology'],
                                                       train_states['op_set_indices'],
                                                       train_states['weights'],
                                                       params,
                                                       block_names,
                                                       train_func,
                                                       train_data,
                                                       val_func,
                                                       val_data,
                                                       test_func,
                                                       test_data)

    """ write results """

    result = {'measure': measure,
              'history': history,
              'weights': weights}

    with open(os.path.join(path, 'block_update_output.pickle'), 'wb') as fid:
        pickle.dump(result, fid)

    with open(os.path.join(path, 'block_update_finish.txt'), 'a') as fid:
        fid.write('x')


if __name__ == "__main__":
    main(sys.argv[1:])
