#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Runable to perform block update

Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""
from __future__ import print_function

import sys, os

module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if not module_path in sys.path:
    sys.path.append(module_path)


from GOP.utility import gop_utils, misc


try:
    import cPickle as pickle
except ImportError:
    import pickle

def main(argv):
            
    path = os.environ['block_update_path']
    
    with open(os.path.join(path, 'params.pickle'), 'rb') as f:
        params = pickle.load(f)
        
    path = os.path.join(params['tmp_dir'], params['model_name'])
    train_func, train_data, val_func, val_data, test_func, test_data = misc.unpickle_generator(os.path.join(path, 'data.pickle'))
    
    if os.path.exists(os.path.join(path, 'custom_loss.pickle')):
        params['loss'] = misc.unpickle_custom_loss(os.path.join(path, 'custom_loss.pickle'))
    
    if os.path.exists(os.path.join(path, 'custom_metrics.pickle')):
        params['metrics'] = misc.unpickle_custom_metrics(os.path.join(path, 'custom_metrics.pickle'))
        
    if os.path.exists(os.path.join(path, 'special_metrics.pickle')):
        params['special_metrics'] = misc.unpickle_special_metrics(os.path.join(path, 'special_metrics.pickle'))
        
    if os.path.exists(os.path.join(path, 'custom_operators.pickle')):
        params['nodal_set'], params['pool_set'], params['activation_set'] = misc.unpickle_custom_operators(os.path.join(path, 'custom_operators.pickle'))
        
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
    
    result = {'measure':measure,
              'history':history,
              'weights':weights}
    
    with open(os.path.join(path, 'block_update_output.pickle'), 'wb') as fid:
        pickle.dump(result, fid)
    
    
    with open(os.path.join(path, 'block_update_finish.txt'), 'a') as fid:
        fid.write('x')

    
if __name__ == "__main__":
    main(sys.argv[1:])
