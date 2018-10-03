#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import os, sys
import numpy as np
import test_utility as utils
import inspect, shutil
import cPickle as pickle

cwd = os.getcwd()

path = os.path.dirname(os.path.dirname(cwd))


if not path in sys.path:
    sys.path.append(path)
    
from GOP.utility import misc
    

def train_func(data):
    # train_data is [input_dim, output_dim]
    steps = 1000
    def gen():
        while True:
            for i in range(1000):
                yield utils.scale(np.random.rand(32, data[0])), np.random.rand(32, data[1])
    
    return gen(), steps

def val_func(data):
    # train_data is [input_dim, output_dim]
    steps = 1000
    def gen():
        while True:
            for i in range(1000):
                yield utils.scale(np.random.rand(32, data[0])), np.random.rand(32, data[1])
    
    return gen(), steps

def test_func(data):
    # train_data is [input_dim, output_dim]
    steps = 1000
    def gen():
        while True:
            for i in range(1000):
                yield utils.scale(np.random.rand(32, data[0])), np.random.rand(32, data[1])
    
    return gen(), steps


def test_BestModelLogging():
    convergence_measure = 'train_mean_squared_error'
    direction = 'lower'
    special_metrics = [utils.special_metric,]
    train_data = [30,10]
    val_data = [30,10]
    test_data = [30,10]
    
    
    cb = misc.BestModelLogging(convergence_measure,
                              direction,
                              special_metrics,
                              train_func,
                              train_data,
                              None,
                              None,
                              test_func,
                              test_data)
    
    assert cb.val_gen is None
    assert cb.val_steps is None
    assert cb.test_gen is not None
    assert isinstance(cb.test_steps, int)
    
    model = utils.get_hemlgop_model(30,10)
    model.compile('adam','mse', ['mse',])
    
    train_gen, train_steps = train_func(train_data)
    model.fit_generator(train_gen, train_steps,
                        epochs=2,
                        callbacks=[cb,])
    
    assert len(cb.performance['train_mean_squared_error']) != 2
    assert len(cb.performance['test_mean_squared_error']) == 2
    assert len(cb.performance['train_special_metric']) == 2
    assert len(cb.performance['test_special_metric']) == 2
    
    assert cb.measure == np.min(cb.performance['train_mean_squared_error'])
    
    convergence_measure = 'val_special_metric'
    cb = misc.BestModelLogging(convergence_measure,
                              direction,
                              special_metrics,
                              train_func,
                              train_data,
                              val_func,
                              val_data,
                              test_func,
                              test_data)
    
    
    val_gen, val_steps = val_func(val_data)
    model.fit_generator(train_gen, train_steps,
                        epochs=2,
                        callbacks=[cb,],
                        validation_data=val_gen,
                        validation_steps=val_steps)
    
    assert cb.measure == np.min(cb.performance['val_special_metric'])
    
    return


def test_initialize_states():
    cwd = os.getcwd()
    test_dir = os.path.join(cwd, 'test_initialize_states') 
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    else:
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
    
    params = {'tmp_dir': cwd,
              'model_name': 'test_initialize_states',
              'use_bias':False,
              'output_activation':'softmax',
              'input_dim':10}
    
    model = 'x'
    train_states = misc.initialize_states(params, model)
    assert train_states == {"layer_iter":   0,
                            "block_iter":   0,
                            "op_sets":       {},
                            "weights":      {},
                            'use_bias':     False,
                            'output_activation': 'softmax',
                            "topology":     [10,],
                            "history":      [],
                            'measure':      [],
                            "is_finished":  False,
                            'model':   model}
    
    
    with open(os.path.join(cwd, 'test_initialize_states', 'train_states.pickle'), 'wa') as fid:
        pickle.dump(train_states, fid)
        
    assert train_states == misc.initialize_states(params, model)
    
    shutil.rmtree(test_dir)
    
    return 

def test_pickle_generator():
    cwd = os.getcwd()
    test_dir = os.path.join(cwd, 'test_pickle_generator') 
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    else:
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
        
    filename = os.path.join(test_dir,'data.pickle')
    
    misc.pickle_generator(filename, train_func, None, None, None, test_func, None)
    
    assert os.path.exists(filename)
    
    train_func_, train_data_, val_func_, val_data_, test_func_, test_data_ = misc.unpickle_generator(filename)
    
    assert train_data_ is None
    assert val_func_ is None
    assert test_func_ is not None
    
    train_gen, train_steps = train_func_([30,10])
    for i in range(10):
        x,y = train_gen.next()
        assert x.shape == (32,30)
        assert y.shape == (32,10)
        
    shutil.rmtree(test_dir)
    return

def test_get_op_set_index():
    start_idx, stop_idx = misc.get_op_set_index(72, 5, 0)
    assert start_idx == 0
    assert stop_idx == 15
    
    start_idx, stop_idx = misc.get_op_set_index(72, 5, 1)
    
    assert start_idx == 15
    assert stop_idx == 30
    
    start_idx, stop_idx = misc.get_op_set_index(72, 5, 2)
    
    assert start_idx == 30
    assert stop_idx == 45  
    
    start_idx, stop_idx = misc.get_op_set_index(72, 5, 3)
    
    assert start_idx == 45
    assert stop_idx == 60
    
    start_idx, stop_idx = misc.get_op_set_index(72, 5, 4)
    
    assert start_idx == 60
    assert stop_idx == 72
    
    
    start_idx, stop_idx = misc.get_op_set_index(72, 72, 1)
    
    assert start_idx == 1
    assert stop_idx == 2
    
    return

def test_check_convergence():
    old = 0.7
    new = 0.3
    direction = 'higher'
    threshold = 0.01
    assert misc.check_convergence(new, old, direction, threshold) == True
    assert misc.check_convergence(old, new, direction, threshold) == False
    assert misc.check_convergence(old, new, 'lower', threshold) == True
    assert misc.check_convergence(new, old, 'lower', threshold) == False
    
    return

def test_pickle_custom_metrics():
    cwd = os.getcwd()
    test_dir = os.path.join(cwd, 'test_pickle_custom_metrics') 
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    else:
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
    
    filename = os.path.join(test_dir, 'custom_metrics.pickle')
    
    metrics = [utils.tf_mae, 'mse', 'mae', utils.tf.losses.softmax_cross_entropy]
    
    has_custom_metrics = misc.pickle_custom_metrics(metrics, filename)
    assert has_custom_metrics
    assert os.path.exists(filename)
    
    metrics = misc.unpickle_custom_metrics(filename)
    
    model = utils.get_hemlgop_model(30,10)
    model.compile('adam','mse', metrics)
    
    train_gen, train_steps = train_func([30,10])
    model.fit_generator(train_gen,
                        train_steps,
                        epochs=2)
    
    shutil.rmtree(test_dir)
    return

def test_pickle_special_metrics():
    cwd = os.getcwd()
    test_dir = os.path.join(cwd, 'test_pickle_special_metrics') 
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    else:
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
    
    filename = os.path.join(test_dir, 'special_metrics.pickle')   
    special_metrics = [utils.acc,]
    
    has_special_metrics = misc.pickle_special_metrics(None, filename)
    assert not has_special_metrics
    has_special_metrics = misc.pickle_special_metrics(special_metrics, filename)
    
    assert has_special_metrics
    assert os.path.exists(filename)
    
    special_metrics = misc.unpickle_special_metrics(filename)
    
    model = utils.get_hemlgop_model(30,10)
    model.compile('adam','mse', ['mae','mse'])
    
    cb = misc.BestModelLogging('val_mean_squared_error',
                              'lower',
                              special_metrics,
                              train_func,
                              [30,10],
                              val_func,
                              [30,10],
                              test_func,
                              [30,10])
    
    train_gen, train_steps = train_func([30,10])
    val_gen, val_steps = val_func([30,10])
    
    model.fit_generator(train_gen,
                        train_steps,
                        epochs=2,
                        callbacks=[cb,],
                        validation_data=val_gen,
                        validation_steps=val_steps)
    
    assert len(cb.performance['val_acc']) == 2
    assert len(cb.performance['test_acc']) == 2
    shutil.rmtree(test_dir)
    
    return



    
    
    