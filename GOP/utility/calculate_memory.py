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
            
    path = os.environ['memory_block_path']
    
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
        
    
    pre_bn_weight, projection, post_bn_weight = gop_utils.calculate_memory_block(params,
                                                                                 train_states,
                                                                                 train_func,
                                                                                 train_data)
    

            
    """ write results """
    
    result = {'pre_bn_weight':pre_bn_weight,
              'projection':projection,
              'post_bn_weight':post_bn_weight}
    
    with open(os.path.join(path, 'memory_block.pickle'), 'wb') as fid:
        pickle.dump(result, fid)
    
    
    with open(os.path.join(path, 'memory_block_finish.txt'), 'a') as fid:
        fid.write('x')

    
if __name__ == "__main__":
    main(sys.argv[1:])
