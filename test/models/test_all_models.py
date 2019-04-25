#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test different model with different parameter settings and different computation environments


Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import model_utility as utility
import pytest
import os
import shutil
import numpy as np
import random
from GOP.models import HeMLGOP, HoMLGOP, HeMLRN, HoMLRN, POP, POPfast, POPmemO, POPmemH

MODELS = [HeMLGOP, HoMLGOP, HeMLRN, HoMLRN, POP, POPfast, POPmemO, POPmemH]
NAMES = ['HeMLGOP', 'HoMLGOP', 'HeMLRN', 'HoMLRN', 'POP', 'POPfast', 'POPmemO', 'POPmemH']
INPUT_DIM = 10
OUTPUT_DIM = 3
BATCH_SIZE = 16
STEPS = 4


def randomized_based_models(tmp_dir, model_name):
    model_path = os.path.join(tmp_dir, 'test_model')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)

    train_seed = np.random.randint(0, 1000)
    val_seed = np.random.randint(0, 1000)
    test_seed = np.random.randint(0, 1000)

    train_func = utility.get_generator
    train_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS, train_seed]
    val_func = utility.get_generator
    val_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS, val_seed]
    test_func = utility.get_generator
    test_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS, test_seed]

    model = MODELS[NAMES.index(model_name)]()

    params = model.get_default_parameters()
    params['tmp_dir'] = tmp_dir
    params['model_name'] = model_name
    params['input_dim'] = INPUT_DIM
    params['output_dim'] = OUTPUT_DIM
    params['loss'] = utility.mean_absolute_error_keras
    params['metrics'] = ['mse', utility.mean_absolute_error_keras]
    params['special_metrics'] = [utility.mean_absolute_error_numpy, ]
    params['convergence_measure'] = 'mean_absolute_error_keras'
    params['direction'] = 'lower'
    params['search_computation'] = ('cpu', 2)
    params['finetune_computation'] = ('cpu', 1)
    params['block_size'] = 3
    params['max_block'] = 2
    params['max_layer'] = 2
    params['nodal_set'] = utility.get_nodal_set()
    params['pool_set'] = utility.get_pool_set()
    params['activation_set'] = utility.get_activation_set()
    params['lr_train'] = (1e-3, 1e-4)
    params['epoch_train'] = (1, 1)
    params['lr_finetune'] = (1e-3, 1e-4)
    params['epoch_finetune'] = (1, 1)
    params['optimizer'] = random.choice(
        ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])
    params['optimizer_parameters'] = random.choice([None, {'lr': 1e-2}])

    model.fit(params,
              train_func,
              train_data,
              val_func,
              val_data,
              test_func,
              test_data)

    test_func = utility.get_test_generator
    test_data_eval = [np.random.rand(BATCH_SIZE * STEPS, INPUT_DIM),
                      np.random.rand(BATCH_SIZE * STEPS, OUTPUT_DIM), BATCH_SIZE]
    test_data_pred = [np.random.rand(BATCH_SIZE * STEPS, INPUT_DIM), None, BATCH_SIZE]

    test_performance_bef = model.evaluate(
        test_func, test_data_eval, params['metrics'], params['special_metrics'], params['finetune_computation'])
    test_pred_bef = model.predict(test_func, test_data_pred, params['finetune_computation'])

    model.save(os.path.join(model_path, model_name + '_pretrained.pickle'))
    model = MODELS[NAMES.index(model_name)]()
    model.load(os.path.join(model_path, model_name + '_pretrained.pickle'))

    test_performance_af = model.evaluate(
        test_func, test_data_eval, params['metrics'], params['special_metrics'], params['finetune_computation'])
    test_pred_af = model.predict(test_func, test_data_pred, params['finetune_computation'])

    assert np.allclose(test_pred_bef, test_pred_af)
    for metric in test_performance_bef.keys():
        assert np.allclose(test_performance_bef[metric], test_performance_af[metric])

    shutil.rmtree(model_path)


def backprop_based_models(tmp_dir, model_name, mem_type=None):
    model_path = os.path.join(tmp_dir, 'test_model')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)

    train_seed = np.random.randint(0, 1000)
    val_seed = np.random.randint(0, 1000)
    test_seed = np.random.randint(0, 1000)

    train_func = utility.get_generator
    train_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS, train_seed]
    val_func = utility.get_generator
    val_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS, val_seed]
    test_func = utility.get_generator
    test_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS, test_seed]

    model = MODELS[NAMES.index(model_name)]()

    params = model.get_default_parameters()
    params['tmp_dir'] = tmp_dir
    params['model_name'] = model_name
    params['input_dim'] = INPUT_DIM
    params['output_dim'] = OUTPUT_DIM
    params['loss'] = utility.mean_absolute_error_keras
    params['metrics'] = ['mse', utility.mean_absolute_error_keras]
    params['special_metrics'] = [utility.mean_absolute_error_numpy, ]
    params['convergence_measure'] = 'mean_absolute_error_keras'
    params['direction'] = 'lower'
    params['search_computation'] = ('cpu', 2)
    params['finetune_computation'] = ('cpu', 1)
    params['max_topology'] = [4, ]
    params['nodal_set'] = utility.get_nodal_set()
    params['pool_set'] = utility.get_pool_set()
    params['activation_set'] = utility.get_activation_set()
    params['lr_train'] = (1e-3, 1e-4)
    params['epoch_train'] = (1, 1)
    params['lr_finetune'] = (1e-3, 1e-4)
    params['epoch_finetune'] = (1, 1)
    params['memory_type'] = mem_type
    params['memory_regularizer'] = 1e-1
    params['optimizer'] = random.choice(
        ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])
    params['optimizer_parameters'] = random.choice([None, {'lr': 1e-2}])

    model.fit(params,
              train_func,
              train_data,
              val_func,
              val_data,
              test_func,
              test_data)

    test_func = utility.get_test_generator
    test_data_eval = [np.random.rand(BATCH_SIZE * STEPS, INPUT_DIM),
                      np.random.rand(BATCH_SIZE * STEPS, OUTPUT_DIM), BATCH_SIZE]
    test_data_pred = [np.random.rand(BATCH_SIZE * STEPS, INPUT_DIM), None, BATCH_SIZE]

    test_performance_bef = model.evaluate(
        test_func, test_data_eval, params['metrics'], params['special_metrics'], params['finetune_computation'])
    test_pred_bef = model.predict(test_func, test_data_pred, params['finetune_computation'])

    model.save(os.path.join(model_path, model_name + '_pretrained.pickle'))
    model = MODELS[NAMES.index(model_name)]()
    model.load(os.path.join(model_path, model_name + '_pretrained.pickle'))

    test_performance_af = model.evaluate(
        test_func, test_data_eval, params['metrics'], params['special_metrics'], params['finetune_computation'])
    test_pred_af = model.predict(test_func, test_data_pred, params['finetune_computation'])

    assert np.allclose(test_pred_bef, test_pred_af)
    for metric in test_performance_bef.keys():
        assert np.allclose(test_performance_bef[metric], test_performance_af[metric])

    shutil.rmtree(model_path)


def test_randomized_based_models(tmpdir):
    randomized_based_models(tmpdir.dirname, 'HeMLGOP')
    randomized_based_models(tmpdir.dirname, 'HoMLGOP')
    randomized_based_models(tmpdir.dirname, 'HeMLRN')
    randomized_based_models(tmpdir.dirname, 'HoMLRN')


def test_backprop_based_models(tmpdir):
    backprop_based_models(tmpdir.dirname, 'POP')
    backprop_based_models(tmpdir.dirname, 'POPfast')
    backprop_based_models(tmpdir.dirname, 'POPmemO', 'PCA')
    backprop_based_models(tmpdir.dirname, 'POPmemO', 'LDA')
    backprop_based_models(tmpdir.dirname, 'POPmemH', 'PCA')
    backprop_based_models(tmpdir.dirname, 'POPmemH', 'LDA')


if __name__ == '__main__':
    pytest.main([__file__])
