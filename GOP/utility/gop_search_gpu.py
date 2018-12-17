#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Runable to perform operator set search procedure on local GPUs

Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""
from __future__ import print_function

import sys, getopt, os

module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

if not module_path in sys.path:
    sys.path.append(module_path)

from GOP.utility import gop_utils, misc

try:
    import cPickle as pickle
except ImportError:
    import pickle

def main(argv):

    try:
      opts, args = getopt.getopt(argv,"b:e:")
    except getopt.GetoptError:
        print('gop_search_gpu.py -b <start index in operator sets> -e <stop index in operator sets>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-b':
            start_idx = int(arg)
        if opt == '-e':
            stop_idx = int(arg)
    
    path = os.environ['gop_search_path']
    machine_no = int(os.environ['gop_machine_no'])
    gpu_no = int(os.environ['CUDA_VISIBLE_DEVICES'])
    
    with open(os.path.join(path, 'params.pickle'), 'rb') as f:
        params = pickle.load(f)
    
    with open(os.path.join(path, 'train_states_tmp.pickle'),'rb') as f:
        train_states = pickle.load(f)
            

    if train_states['model'] in ['HeMLGOP', 'HoMLGOP', 'HeMLRN', 'HoMLRN']:
        search_routine = gop_utils.IRS
    elif train_states['model'] in ['POPfast', 'POPmemO', 'POPmemH']:
        search_routine = gop_utils.GISfast
    elif train_states['model'] == 'POP':
        search_routine = gop_utils.GIS
    else:
        raise Exception('The given model "%s" is not supported by the operator set search procedure' % train_states['model'])
    
    performance, weights, op_set_idx, history = search_routine(start_idx,
                                                            stop_idx,
                                                            params, 
                                                            train_states)
    
    result = {'performance':performance,
              'weights':weights,
              'op_set_idx':op_set_idx,
              'history': history}
    
    with open(os.path.join(path, 'machine_%d_%d.pickle' % (machine_no, gpu_no)), 'wb') as fid:
        pickle.dump(result, fid)
    
    
    with open(os.path.join(path, 'machine_%d_result.txt' % machine_no), 'a') as fid:
        fid.write('%d, %.4f \n' % (gpu_no, performance[params['convergence_measure']]))
    
    
    with open(os.path.join(path, 'machine_%d_%d_finish.txt' % (machine_no, gpu_no)), 'w') as fid:
        fid.write('x')
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
