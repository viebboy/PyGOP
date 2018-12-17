#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Fast Progressive Operational Perceptron with memory
https://arxiv.org/abs/1808.06377


Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

from __future__ import print_function

from ..utility import misc, gop_utils, gop_operators
from ._model import _Model
import shutil, os
try:
    import cPickle as pickle
except ImportError:
    import pickle
    


    
class POPmemO(_Model):
    """Progressive Operational Perceptron with Memory scheme O 
    
    This class implements the POPmemH algorithm to learn a multilayer 
    network of Generalized Operational Perceptron (GOP) in a progressive manner
    augmented by memory scheme O. The operator set search procedure is similar
    to POPfast. Difference between POPmemO and POPmemH is described in the 
    original paper in the reference. 
    
    reference: https://arxiv.org/abs/1808.06377
    
    Note:
        The model basically uses a python data generator mechanism to feed the data
        Thus, the data should be prepared with the following format:
            train_func: a function that returns (data_generator, number of steps)
                        the user specifies how data is loaded and preprocess by giving
                        the definition of the data generator and the number of steps (number of mini-batches)
            train_data: the input to 'train_func'
                        this can be filepath to the data on disk
            
        When the model generates the data, the generator and #steps are retrieved by
        calling:
            gen, steps = train_func(train_data)
            
        And next(gen) should produce a mini-batch of (x,y) in case of fit()/evaluate() 
        or only x in case of predict()
        
        See documentation page for example how to define such train_func and train_data
        
        
    Examples:
        >>> from GOP import models
        >>> model = models.POPmemO() # create a model instance
        >>> params = model.get_default_parameters() # get default parameters (PCA as memory path)
        
        # fit the model with data using train_func, train_data (see above in Note)
        >>> performance, progressive_history, finetune_history = model.fit(params, train_func, train_data)
        
        # evaluate model with test data (test_func, test_data) using a list of metrics
        # and special_metrics (those require full batch evaluation such as F1, Precision, Recall...)
        # using either CPU or GPU as computation environment,
        # e.g. computation=('cpu',) -> using cpu
        # e.g. computation=('gpu', [0,1]) -> using gpu with GPU0, GPU1
        
        >>> model.evaluate(test_func, test_data, metrics, special_metrics, computation) 

        # generate prediction with (test_func, test_data)
        >>> model.predict(test_func, test_data, computation)
        
        # save model to 'popmemo.pickle'
        >>> model.save('popmemo.pickle')
        
        # load model from 'popmemo.pickle' and finetune with another data and
        # (possibly different) parameter settings
        >>> model.load('popmemo.pickle')
        >>> history, performance = model.finetune(params, another_train_func, another_train_data)
        
        See documentation page for detail usage and explanation
        
        
    """    
    def __init__(self, *args, **kargs):
        self.model_data = None 
        self.model_data_attributes = ['model', 
                                      'topology', 
                                      'op_sets', 
                                      'weights', 
                                      'output_activation', 
                                      'use_bias']
        
        self.model_name = 'POPmemO'
        
        return
    
    def get_default_parameters(self,):
        
        params = {}
        
        params['memory_regularizer'] = 0.1
        params['memory_type'] = 'PCA'
        params['min_energy_percentage'] = 0.98
        params['layer_threshold'] = 1e-4
        params['metrics'] = ['mse',]
        params['convergence_measure'] = 'mse'
        params['direction'] = 'lower'
        params['output_activation'] = None
        params['loss'] = 'mse'
        params['max_topology'] = [40, 40, 40, 40]
        params['weight_regularizer'] = None
        params['weight_regularizer_finetune'] = None
        params['weight_constraint'] = 2.0
        params['weight_constraint_finetune'] = 2.0
        params['lr_train'] = (0.01,0.001,0.0001)
        params['epoch_train'] = (2,2,2)
        params['lr_finetune'] = (0.0005,)
        params['epoch_finetune'] = (2,)
        params['input_dropout'] = None
        params['dropout'] = 0.2
        params['dropout_finetune'] = 0.2
        params['optimizer'] = 'adam'
        params['nodal_set'] = gop_operators.get_default_nodal_set()
        params['pool_set'] = gop_operators.get_default_pool_set()
        params['activation_set'] = gop_operators.get_default_activation_set()
        params['cluster'] = False
        params['special_metrics'] = None
        params['search_computation'] = ('cpu', 8)
        params['finetune_computation'] = ('cpu', 8)
        params['use_bias'] = True
        params['direct_computation'] = False
        params['class_weight'] = None
        
        return  params
    
    def check_parameters(self, params):
        
        if 'memory_type' in params.keys():
            assert params['memory_type'] in ['PCA', 'LDA'], 'Only support 2 memory types "PCA" and "LDA", given %s' % params['memory_type']
        
        if params['memory_type'] == 'PCA':
            if 'min_energy_percentage' in params.keys():
                assert params['min_energy_percentage'] > 0 and params['min_energy_percentage'] < 1.0, 'Minimum energy to keep in PCA must be nonzero and less than 1, given %s' % str(params['min_energy_percentage'])
        
        
        params = misc.check_model_parameters(params, self.get_default_parameters())
        
        return params
    
    def fit(self, 
            params,
            train_func, 
            train_data, 
            val_func=None, 
            val_data=None, 
            test_func=None, 
            test_data=None,
            verbose=False):
        
        if verbose:
            print('Start progressive learning')
        
        p_history = self.progressive_learn(params,
                                              train_func,
                                              train_data,
                                              val_func,
                                              val_data,
                                              test_func,
                                              test_data,
                                              verbose)
        
        if verbose:
            print('Start finetuning')
        
        f_history, performance = self.finetune(params,
                                               train_func,
                                               train_data,
                                               val_func,
                                               val_data,
                                               test_func,
                                               test_data,
                                               verbose)
        
        return performance, p_history, f_history
        
        
        
    def progressive_learn(self,
                          params,
                          train_func, 
                          train_data,
                          val_func=None, 
                          val_data=None,
                          test_func=None,
                          test_data=None,
                          verbose=False):
    
        params = self.check_parameters(params)

        original_convergence_measure = params['convergence_measure']        
        if val_func:
            params['convergence_measure'] = 'val_' + params['convergence_measure'] 
        else:
            params['convergence_measure'] = 'train_' + params['convergence_measure']
        
        
        misc.test_generator(train_func, train_data, params['input_dim'], params['output_dim'])
        if val_func:    misc.test_generator(val_func, val_data, params['input_dim'], params['output_dim'])
        if test_func:   misc.test_generator(test_func, test_data, params['input_dim'], params['output_dim'])
        
        train_states = misc.initialize_states(params, self.model_name)
        train_states['block_iter'] = 1
        
        if not train_states['is_finished']:
            for layer_iter in range(train_states['layer_iter'], len(params['max_topology'])):
                
                if verbose:
                    print('-------------Layer %d ------------------' %layer_iter)
                    print('##### Calculating memory block ########')
                    
                pre_bn_weight, projection, post_bn_weight = gop_utils.calculate_memory_block_standalone(params,
                                                                                                        train_states,
                                                                                                        train_func,
                                                                                                        train_data,
                                                                                                        val_func,
                                                                                                        val_data,
                                                                                                        test_func,
                                                                                                        test_data)
                
                
                if layer_iter > 0:
                    train_states['topology'].pop()
                    
                train_states['topology'].append([('mem', projection[0].shape[1])])
                train_states['topology'].append(('dense', params['output_dim']))
                
                suffix = '_' + str(layer_iter) + '_0'
                train_states['weights']['mem_pre_bn' + suffix] = pre_bn_weight
                train_states['weights']['mem' + suffix] = projection
                train_states['weights']['bn' + suffix] = post_bn_weight
                
                train_states['history'].append([])
                train_states['measure'].append([])
                
                        
                if verbose:
                    print('##### GISfast #####')
                    
                if params['cluster']:
                    search_routine = gop_utils.search_cluster
                else:
                    if params['search_computation'][0] == 'cpu':
                        search_routine = gop_utils.search_cpu
                    else:
                        search_routine = gop_utils.search_gpu
                    
                              
                block_performance, block_weights, block_op_set_idx, history = search_routine(params,
                                                                                          train_states,
                                                                                          train_func,
                                                                                          train_data,
                                                                                          val_func,
                                                                                          val_data,
                                                                                          test_func,
                                                                                          test_data)
                if verbose:
                    self.print_performance(history, params['convergence_measure'], params['direction'])
                    
                    
                train_states['measure'][layer_iter].append(block_performance[params['convergence_measure']])
                train_states['history'][layer_iter].append(history)
                    
                if layer_iter > 0:
                    if misc.check_convergence(train_states['measure'][layer_iter][-1], train_states['measure'][layer_iter-1][-1], params['direction'], params['layer_threshold']):
                        train_states['topology'].pop(-2)
                        
                        suffix = '_' + str(layer_iter) + '_0'
                        del train_states['weights']['mem_pre_bn' + suffix]
                        del train_states['weights']['mem' + suffix]
                        del train_states['weights']['bn' + suffix]
                        break
                        
                suffix = '_' + str(layer_iter) + '_1'
                train_states['topology'][-2].append(('gop', params['max_topology'][layer_iter]))
                train_states['op_set_indices']['gop' + suffix] = block_op_set_idx
                train_states['weights']['gop' + suffix], train_states['weights']['bn'+suffix], train_states['weights']['output'] = block_weights
                train_states['layer_iter'] += 1
                
                
                    
            train_states['is_finished'] = True
            path = os.path.join(params['tmp_dir'], params['model_name'], 'train_states.pickle')
            with open(path, 'wb') as fid:
                pickle.dump(train_states, fid)             
            
            if os.path.exists(os.path.join(params['tmp_dir'], params['model_name'])):
                shutil.rmtree(os.path.join(params['tmp_dir'], params['model_name']))
            

        model_data = {'model': self.model_name,
                      'topology': train_states['topology'],
                      'op_sets': misc.map_operator_from_index(train_states['op_set_indices'], params['nodal_set'], params['pool_set'], params['activation_set']),
                      'weights': train_states['weights'],
                      'use_bias': train_states['use_bias'],
                      'output_activation': train_states['output_activation']}
        
        self.model_data = model_data
        params['convergence_measure'] = original_convergence_measure
        
        return train_states['history']
                
