#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""
import test_utility
import os, sys
import numpy as np
import utils
import inspect

cwd = os.getcwd()

path = os.path.dirname(os.path.dirname(cwd))
print(path)

if not path in sys.path:
    sys.path.append(path)

from GOP.utility import gop_utils, misc, gop_operators


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


def test_network_builder():
    
    print('testing network_builder()')
    topology = [30, [('gop',20)], ('dense',10)]
    op_sets = {'gop_0_0': gop_operators.get_random_op_set()}
    
    model = gop_utils.network_builder(topology, op_sets)
    
    model.compile('adam','mse', ['mse',])
    
    model.summary()
    
    print('finish testing network_builder()')
    return
    
def test_network_trainer():
    
    print('testing network_trainer()')
    topology = [30, [('gop',20)], ('dense',10)]
    op_sets = {'gop_0_0': gop_operators.get_random_op_set()}

    model = gop_utils.network_builder(topology, op_sets)
    
    model.compile('adam','mse', ['mse',])
    
    direction = 'lower'
    convergence_measure = 'val_mae'
    LR = (0.01,0.001)
    SC = (2,2)
    optimizer = 'adam'
    loss = 'mse'
    metrics = ['mse',]
    special_metrics = [test_utility.tf_mae,]
    train_data = (30,10)
    val_data = (30,10)
    test_data = (30,10)
    
    train_gen, train_steps = train_func(train_data)
    val_gen, val_steps = val_func(val_data)
    
    cb = misc.BestModelLogging(convergence_measure, 
                                   direction,
                                   special_metrics,
                                   train_func,
                                   train_data,
                                   val_func,
                                   val_data,
                                   test_func,
                                   test_data)
    
    model.fit_generator(train_gen, 
                        train_steps, 
                        epochs=1, 
                        callbacks=[cb,],
                        validation_data=val_gen,
                        validation_steps=val_steps)
    
    
    measure, history, weights = gop_utils.network_trainer(model,
                                        direction,
                                        convergence_measure,
                                        LR,
                                        SC,
                                        optimizer,
                                        loss,
                                        metrics,
                                        special_metrics,
                                        train_func,
                                        train_data,
                                        val_func,
                                        val_data,
                                        test_func,
                                        test_data)
    
    true_measure = np.min(history[convergence_measure])
    assert measure == true_measure, 'returned measure (%.4f) != min %s (%.4f)' %(measure, convergence_measure, true_measure)
    for metric in history.keys():
        assert len(history[metric]) == 4, 'length of %s should be 4'% metric
    print('finish testing network_trainer()')
    
    return

