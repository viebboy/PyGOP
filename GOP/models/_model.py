#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy

Abstract base model for all progressive learning models
"""
from __future__ import print_function

from ..utility import misc, gop_utils
import os, dill, numpy as np
dill.settings['recurse'] = True
    
CUDA_FLAG = 'CUDA_VISIBLE_DEVICES'


class _Model:
    
    """Abstract base model for all progressive learning models (private, used as base class for implementation)
    
    This base class implements common methods for all models, including 
    HeMLGOP, HoMLGOP, HeMLRN, HoMLRN, POP, POPfast, POPmemO, POPmemH
    
    Common methods include
    save(), load(), check_parameters(), finetune(), evaluate(), predict()
    
    Derived class should implement
    fit(), progressive_learn(), get_default_parameters() and possibly overwrite check_parameters()
    
    """
    def __init__(self, *args, **kargs):
        return

    
    def save(self, filename):
        if self.model_data is None:
            raise Exception('No existing model data to save')
        else:
            with open(filename,'wb') as fid:
                dill.dump(self.model_data, fid, recurse=True)
        
        return
    
    def load(self, filename):
        
        self.model_data = gop_utils.load(filename, self.model_data_attributes, self.model_name)
        
        return
    
    def get_default_parameters(self,):
        return {}
        
    def check_parameters(self, params):
        
        params = misc.check_model_parameters(params, self.get_default_parameters())
        
        return params

    def finetune(self, 
                 params,
                 train_func, 
                 train_data,
                 val_func=None, 
                 val_data=None,
                 test_func=None,
                 test_data=None,
                 verbose=False):

            
        params = self.check_parameters(params)
        
        if CUDA_FLAG in os.environ.keys():
            cuda_status = os.environ[CUDA_FLAG]
        else:
            cuda_status = None
            
        if params['finetune_computation'][0] == 'cpu':
            os.environ[CUDA_FLAG] = ''
        else:
            os.environ[CUDA_FLAG] = misc.get_gpu_str(params['finetune_computation'][1])
            
        
        try:
            history, performance, self.model_data = gop_utils.finetune(self.model_data, 
                                                                         params,
                                                                         train_func,
                                                                         train_data,
                                                                         val_func,
                                                                         val_data,
                                                                         test_func,
                                                                         test_data)
        except:
            if cuda_status is None:
                del os.environ[CUDA_FLAG]
            else:
                os.environ[CUDA_FLAG] = cuda_status
                
            raise Exception('Failed to finetune the model')
        
        if cuda_status is None:
            del os.environ[CUDA_FLAG]
        else:
            os.environ[CUDA_FLAG] = cuda_status
            
        if val_func:
            convergence_measure = 'val_' + params['convergence_measure'] 
        else:
            convergence_measure = 'train_' + params['convergence_measure']
            
        if verbose:
            self.print_performance(history, convergence_measure, params['direction'])
            
        
        return history, performance
        
    def evaluate(self, 
                 data_func, 
                 data_argument, 
                 metrics, 
                 special_metrics=None, 
                 computation=('cpu',)):

        if CUDA_FLAG in os.environ.keys():
            cuda_status = os.environ[CUDA_FLAG]
        else:
            cuda_status = None
            
        if computation[0] == 'cpu':
            os.environ[CUDA_FLAG] = ''
        else:
            os.environ[CUDA_FLAG] = misc.get_gpu_str(computation[1])
            
        
        try:
            performance = gop_utils.evaluate(self.model_data,
                                             data_func,
                                             data_argument,
                                             metrics,
                                             special_metrics)
        
        except:
            if cuda_status is None:
                del os.environ[CUDA_FLAG]
            else:
                os.environ[CUDA_FLAG] = cuda_status
                
            raise Exception('Failed to evaluate the model')
        
        if cuda_status is None:
            del os.environ[CUDA_FLAG]
        else:
            os.environ[CUDA_FLAG] = cuda_status
        
        return performance
        
    
    def predict(self, data_func, data_argument, computation=('cpu',)):
        
        if CUDA_FLAG in os.environ.keys():
            cuda_status = os.environ[CUDA_FLAG]
        else:
            cuda_status = None
            
        if computation[0] == 'cpu':
            os.environ[CUDA_FLAG] = ''
        else:
            os.environ[CUDA_FLAG] = misc.get_gpu_str(computation[1])
            
        try:
            pred = gop_utils.predict(self.model_data,
                                     data_func,
                                     data_argument)
        except:
            if cuda_status is None:
                del os.environ[CUDA_FLAG]
            else:
                os.environ[CUDA_FLAG] = cuda_status
                
            raise Exception('Failed to use the model to predict')
        
        if cuda_status is None:
            del os.environ[CUDA_FLAG]
        else:
            os.environ[CUDA_FLAG] = cuda_status
        
        return pred
    
    def print_performance(self, history, convergence_measure, direction):
        if isinstance(history[convergence_measure], list):
            idx = np.argmin(history[convergence_measure]) if direction == 'lower' else np.argmax(history[convergence_measure])
        
            for metric in history.keys():
                if history[metric] is not None:
                    print('%s: %.4f' % (metric, history[metric][idx]))
        else:
            for metric in history.keys():
                if history[metric] is not None:
                    print('%s: %.4f' % (metric, history[metric]))
        return
