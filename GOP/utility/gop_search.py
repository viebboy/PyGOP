#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Runable to perform operator set search procedure

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
      opts, args = getopt.getopt(argv,"N:n:")
    except getopt.GetoptError:
        print('gop_search.py -N <number of machine> -n <machine index>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-N':
            no_machine = int(arg)
        if opt == '-n':
            machine_no = int(arg)
            
    path = os.environ['gop_search_path']
    
    with open(os.path.join(path, 'params.pickle'), 'rb') as f:
        params = pickle.load(f)
        
    no_op_set = params['no_op_set']
    
    start_idx, stop_idx = misc.get_op_set_index(no_op_set, no_machine, machine_no)
    
    if params['search_computation'][0] == 'cpu':
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
                    
    if params['search_computation'][0] == 'gpu':
            
        performance, weights, op_set_idx, history = gop_utils.search_gpu_(path,
                                                                     params,
                                                                     start_idx,
                                                                     stop_idx,
                                                                     machine_no)

            
    """ write results """
    
    result = {'performance':performance,
              'weights':weights,
              'op_set_idx':op_set_idx,
              'history': history}
    
    with open(os.path.join(path, 'machine_%d.pickle' % machine_no), 'wb') as fid:
        pickle.dump(result, fid)
    
    
    with open(os.path.join(path, 'machine_result.txt'), 'a') as fid:
        fid.write('%d, %.4f \n' % (machine_no, performance[params['convergence_measure']]))
    
    
    with open(os.path.join(path, 'machine_%d_finish.txt' % machine_no), 'w') as fid:
        fid.write('x')
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
