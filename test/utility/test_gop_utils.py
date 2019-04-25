#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import pytest
import os
import pickle
import dill
import shutil
import random
import utility
import numpy as np
from GOP.utility import gop_utils, gop_operators, misc

INPUT_DIM = utility.INPUT_DIM
OUTPUT_DIM = utility.OUTPUT_DIM
BATCH_SIZE = 32
STEPS = 4


def test_he_init():
    assert gop_utils.he_init(fan_in=10, fan_out=10, shape=(10, 20)).shape == (10, 20)


def test_get_random_gop_weight():
    weights = gop_utils.get_random_gop_weight(input_dim=10, output_dim=3, use_bias=True)
    assert len(weights) == 2
    assert weights[0].shape == (1, 10, 3)
    assert weights[1].shape == (3,)

    weights = gop_utils.get_random_gop_weight(input_dim=10, output_dim=3, use_bias=False)
    assert len(weights) == 1


def test_get_random_block_weights():
    gop_weights = [np.random.rand(1, 10, 20), np.random.rand(20,)]
    bn_weights = [np.random.rand(20,), ] * 4
    output_weights = [np.random.rand(100, 3), np.random.rand(3,)]

    new_gop_weights, new_bn_weights, new_output_weights = gop_utils.get_random_block_weights(
        [gop_weights, bn_weights, output_weights])
    assert new_gop_weights[0].shape == gop_weights[0].shape
    assert len(new_gop_weights) == 2
    assert new_bn_weights[0].shape == (20,)
    assert new_output_weights[0].shape[0] == 100 + 20


def network_builder():
    params, model_data = utility.get_random_model_data()

    model = gop_utils.network_builder(model_data['topology'],
                                      model_data['op_sets'],
                                      input_dropout=params['input_dropout'],
                                      dropout=params['dropout'],
                                      regularizer=params['weight_regularizer'],
                                      constraint=params['weight_constraint'],
                                      output_activation=model_data['output_activation'],
                                      use_bias=model_data['use_bias'])

    model.compile(params['optimizer'], params['loss'], params['metrics'])

    output_weights = model.get_layer('output').get_weights()
    if output_weights[0].ndim == 3:
        assert output_weights[0].shape == (1, 50, OUTPUT_DIM)
    else:
        assert output_weights[0].shape == (50, OUTPUT_DIM)


def test_network_builder():
    network_builder()
    network_builder()


def network_trainer():
    params, model_data = utility.get_random_model_data()

    model = gop_utils.network_builder(model_data['topology'],
                                      model_data['op_sets'],
                                      input_dropout=params['input_dropout'],
                                      dropout=params['dropout'],
                                      regularizer=params['weight_regularizer'],
                                      constraint=params['weight_constraint'],
                                      output_activation=model_data['output_activation'],
                                      use_bias=model_data['use_bias'])

    model.compile(params['optimizer'], params['loss'], params['metrics'])

    convergence_measure = random.choice(['train_', 'val_']) + params['convergence_measure']

    measure, history, weights = gop_utils.network_trainer(model,
                                                          direction=params['direction'],
                                                          convergence_measure=convergence_measure,
                                                          LR=params['lr_finetune'],
                                                          SC=params['epoch_finetune'],
                                                          optimizer=params['optimizer'],
                                                          optimizer_parameters=params['optimizer_parameters'],
                                                          loss=params['loss'],
                                                          metrics=params['metrics'],
                                                          special_metrics=params['special_metrics'],
                                                          train_func=utility.get_generator,
                                                          train_data=[
                                                              INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                          val_func=utility.get_generator,
                                                          val_data=[INPUT_DIM, OUTPUT_DIM,
                                                                    BATCH_SIZE, STEPS],
                                                          test_func=utility.get_generator,
                                                          test_data=[INPUT_DIM,
                                                                     OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                          class_weight=params['class_weight'])


def test_network_trainer():
    network_trainer()
    network_trainer()


def test_finetune():
    params, model_data = utility.get_random_model_data()
    history, performance, data = gop_utils.finetune(model_data,
                                                    params,
                                                    utility.get_generator,
                                                    [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                    utility.get_generator,
                                                    [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                    utility.get_generator,
                                                    [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])


def test_evaluate():
    params, model_data = utility.get_random_model_data()
    gop_utils.evaluate(model_data=model_data,
                       func=utility.get_generator,
                       data=[INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                       metrics=params['metrics'],
                       special_metrics=params['special_metrics'])


def test_predict():
    params, model_data = utility.get_random_model_data()
    predictions = gop_utils.predict(model_data, func=utility.get_generator, data=[
                                    INPUT_DIM, None, BATCH_SIZE, STEPS])
    assert predictions.shape == (BATCH_SIZE * STEPS, OUTPUT_DIM)


def test_load(tmpdir):
    params, model_data = utility.get_random_model_data()
    model_data_attributes = ['model', 'weights', 'topology',
                             'op_sets', 'output_activation', 'use_bias']
    filename = os.path.join(tmpdir.dirname, 'model_data.pickle')
    with open(filename, 'wb') as fid:
        dill.dump(model_data, fid, recurse=True)

    model_data_recovered = gop_utils.load(filename, model_data_attributes, model_data['model'])
    assert model_data_recovered['topology'] == model_data['topology']
    assert model_data_recovered['op_sets'] == model_data['op_sets']
    layer_name = random.choice(list(model_data['weights'].keys()))
    assert np.allclose(model_data_recovered['weights']
                       [layer_name][0], model_data['weights'][layer_name][0])


def test_PCA():
    params, train_states = utility.get_random_popmem_states()
    gop_utils.PCA(params, train_states, utility.get_generator,
                  [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])


def test_LDA():
    params, train_states = utility.get_random_popmem_states()
    gop_utils.LDA(params, train_states, utility.get_generator,
                  [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])


def test_calculate_memory_block_standalone(tmpdir):
    model_path = os.path.join(tmpdir.dirname, 'test_model')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)

    params, train_states = utility.get_random_popmem_states()
    params['tmp_dir'] = tmpdir.dirname
    params['model_name'] = 'test_model'
    params['search_computation'] = ('cpu', 1)

    params['memory_type'] = 'PCA'
    gop_utils.calculate_memory_block_standalone(params,
                                                train_states,
                                                utility.get_generator,
                                                [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                utility.get_generator,
                                                [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                utility.get_generator,
                                                [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    shutil.rmtree(model_path)
    os.mkdir(model_path)

    params['memory_type'] = 'LDA'
    gop_utils.calculate_memory_block_standalone(params,
                                                train_states,
                                                utility.get_generator,
                                                [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                utility.get_generator,
                                                [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                utility.get_generator,
                                                [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    shutil.rmtree(model_path)


def test_GIS_(tmpdir):
    params, train_states = utility.get_random_states()
    params['convergence_measure'] = random.choice(
        ['train_', 'val_']) + params['convergence_measure']
    params['tmp_dir'] = tmpdir.dirname
    params['model_name'] = 'test_model'
    params['max_topology'] = [20, 20, 20]
    params['layer_iter'] = 1
    train_states['topology'][-1] = ('gop', OUTPUT_DIM)
    no_op_set = len(params['nodal_set']) * len(params['pool_set']) * len(params['activation_set'])

    if os.path.exists(os.path.join(tmpdir.dirname, 'test_model')):
        shutil.rmtree(os.path.join(tmpdir.dirname, 'test_model'))
    os.mkdir(os.path.join(tmpdir.dirname, 'test_model'))

    # perform GIS search on output layer
    train_states['search_layer'] = 'output'
    train_states['hidden_op_set_idx'] = np.random.randint(0, no_op_set)

    misc.dump_data(params,
                   train_states,
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    with open(os.path.join(tmpdir.dirname, 'test_model', 'params.pickle'), 'rb') as fid:
        params = pickle.load(fid)

    op_set_idx = np.random.randint(0, no_op_set)

    gop_utils.GIS_(params, train_states, op_set_idx)

    # perform GIS search on hidden layer
    train_states['search_layer'] = 'hidden'
    train_states['output_op_set_idx'] = np.random.randint(0, no_op_set)

    misc.dump_data(params,
                   train_states,
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    op_set_idx = np.random.randint(0, no_op_set)

    gop_utils.GIS_(params, train_states, op_set_idx)

    shutil.rmtree(os.path.join(tmpdir.dirname, 'test_model'))


def test_GISfast_(tmpdir):
    params, train_states = utility.get_random_states()
    params['convergence_measure'] = random.choice(
        ['train_', 'val_']) + params['convergence_measure']
    params['tmp_dir'] = tmpdir.dirname
    params['model_name'] = 'test_model'
    params['max_topology'] = [20, 20, 20]
    params['layer_iter'] = 1
    params['block_iter'] = 0
    no_op_set = len(params['nodal_set']) * len(params['pool_set']) * len(params['activation_set'])

    if os.path.exists(os.path.join(tmpdir.dirname, 'test_model')):
        shutil.rmtree(os.path.join(tmpdir.dirname, 'test_model'))
    os.mkdir(os.path.join(tmpdir.dirname, 'test_model'))

    misc.dump_data(params,
                   train_states,
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    with open(os.path.join(tmpdir.dirname, 'test_model', 'params.pickle'), 'rb') as fid:
        params = pickle.load(fid)

    op_set_idx = np.random.randint(0, no_op_set)

    gop_utils.GISfast_(params, train_states, op_set_idx)

    shutil.rmtree(os.path.join(tmpdir.dirname, 'test_model'))


def test_LS(tmpdir):
    params, train_states = utility.get_random_states()
    params['convergence_measure'] = random.choice(
        ['train_', 'val_']) + params['convergence_measure']
    params['tmp_dir'] = tmpdir.dirname
    params['model_name'] = 'test_model'

    if os.path.exists(os.path.join(tmpdir.dirname, 'test_model')):
        shutil.rmtree(os.path.join(tmpdir.dirname, 'test_model'))
    os.mkdir(os.path.join(tmpdir.dirname, 'test_model'))

    misc.dump_data(params,
                   train_states,
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                   utility.get_generator,
                   [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    with open(os.path.join(tmpdir.dirname, 'test_model', 'params.pickle'), 'rb') as fid:
        params = pickle.load(fid)

    op_set_idx = np.random.randint(
        0, len(params['nodal_set']) * len(params['pool_set']) * len(params['activation_set']))

    gop_utils.LS(params, train_states, op_set_idx)

    shutil.rmtree(os.path.join(tmpdir.dirname, 'test_model'))


def test_search_cpu(tmpdir):
    model_path = os.path.join(tmpdir.dirname, 'test_model')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)

    # test HeMLGOP, HoMLGOP, HeMLRN, HoMLRN
    params, train_states = utility.get_random_states()

    params['tmp_dir'] = tmpdir.dirname
    params['model_name'] = 'test_model'
    params['convergence_measure'] = random.choice(
        ['train_', 'val_']) + params['convergence_measure']
    params['no_op_set'] = len(params['nodal_set']) * \
        len(params['pool_set']) * len(params['activation_set'])
    params['search_computation'] = ('cpu', 2)
    train_states['model'] = random.choice(['HeMLGOP', 'HoMLGOP', 'HeMLRN', 'HoMLRN'])

    gop_utils.search_cpu(params,
                         train_states,
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    shutil.rmtree(model_path)

    # test POPfast
    params['max_topology'] = [20, 20, 20]
    params['layer_iter'] = 1
    params['block_iter'] = 0
    train_states['model'] = 'POPfast'

    os.mkdir(model_path)
    gop_utils.search_cpu(params,
                         train_states,
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    shutil.rmtree(model_path)

    # test POP
    train_states['model'] = 'POP'
    train_states['topology'][-1] = ('gop', OUTPUT_DIM)
    os.mkdir(model_path)

    # with GIS for output layer
    train_states['search_layer'] = 'output'
    train_states['hidden_op_set_idx'] = np.random.randint(0, params['no_op_set'])

    gop_utils.search_cpu(params,
                         train_states,
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    # with GIS for hidden layer
    train_states['search_layer'] = 'hidden'
    train_states['output_op_set_idx'] = np.random.randint(0, params['no_op_set'])

    gop_utils.search_cpu(params,
                         train_states,
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                         utility.get_generator,
                         [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    shutil.rmtree(model_path)


def test_block_update():
    params, model_data = utility.get_random_model_data()
    params['nodal_set'] = gop_operators.get_default_nodal_set()
    params['pool_set'] = gop_operators.get_default_pool_set()
    params['activation_set'] = gop_operators.get_default_activation_set()
    params['convergence_measure'] = random.choice(
        ['train_', 'val_']) + params['convergence_measure']
    block_names = ['gop_0_0', 'gop_1_0', 'bn_0_0', 'bn_1_0', 'output']

    all_op_sets = utility.get_all_operators()

    op_set_indices = {}
    for layer_name in model_data['op_sets'].keys():
        op_set_indices[layer_name] = all_op_sets.index(model_data['op_sets'][layer_name])

    _, _, new_weights = gop_utils.block_update(model_data['topology'],
                                               op_set_indices,
                                               model_data['weights'],
                                               params,
                                               block_names,
                                               utility.get_generator,
                                               [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                               utility.get_generator,
                                               [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                               utility.get_generator,
                                               [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    for layer_name in new_weights.keys():
        if layer_name not in block_names:
            assert np.allclose(new_weights[layer_name][0], model_data['weights'][layer_name][0])


def test_block_update_standalone(tmpdir):
    model_path = os.path.join(tmpdir.dirname, 'test_model')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)

    params, model_data = utility.get_random_model_data()
    params['tmp_dir'] = tmpdir.dirname
    params['model_name'] = 'test_model'
    params['nodal_set'] = gop_operators.get_default_nodal_set()
    params['pool_set'] = gop_operators.get_default_pool_set()
    params['activation_set'] = gop_operators.get_default_activation_set()
    params['convergence_measure'] = random.choice(
        ['train_', 'val_']) + params['convergence_measure']
    block_names = ['gop_0_0', 'gop_1_0', 'bn_0_0', 'bn_1_0', 'output']

    all_op_sets = utility.get_all_operators()

    op_set_indices = {}
    for layer_name in model_data['op_sets'].keys():
        op_set_indices[layer_name] = all_op_sets.index(model_data['op_sets'][layer_name])

    train_states = {'topology': model_data['topology'],
                    'weights': model_data['weights'],
                    'op_set_indices': op_set_indices}

    _, _, new_weights = gop_utils.block_update_standalone(train_states,
                                                          params,
                                                          block_names,
                                                          utility.get_generator,
                                                          [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                          utility.get_generator,
                                                          [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS],
                                                          utility.get_generator,
                                                          [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    for layer_name in new_weights.keys():
        if layer_name not in block_names:
            assert np.allclose(new_weights[layer_name][0], model_data['weights'][layer_name][0])

    shutil.rmtree(model_path)


def test_get_optimal_op_set():
    search_results = []
    convergence_measure = random.choice(['train_', 'val_']) + \
        random.choice(['mean_squared_error', 'mean_absolute_error'])
    direction = 'lower'
    lowest_value = -np.random.rand(1)[0] - 1.0

    for i in range(10):
        performance = {'train_mean_squared_error': np.random.rand(1)[0],
                       'val_mean_squared_error': np.random.rand(1)[0],
                       'train_mean_absolute_error': np.random.rand(1)[0],
                       'val_mean_absolute_error': np.random.rand(1)[0]}

        weights = None
        op_set_idx = None
        history = None
        search_results.append((performance, weights, op_set_idx, history))

    best_performance = {'train_mean_squared_error': lowest_value,
                        'val_mean_squared_error': lowest_value,
                        'train_mean_absolute_error': lowest_value,
                        'val_mean_absolute_error': lowest_value}

    weights = np.random.rand(10, 20)
    op_set_idx = np.random.randint(0, 1000)

    search_results.append((best_performance, weights, op_set_idx, history))

    recovered_result = gop_utils.get_optimal_op_set(search_results, convergence_measure, direction)

    assert recovered_result[0] == best_performance
    assert np.allclose(recovered_result[1], weights)
    assert recovered_result[2] == op_set_idx


if __name__ == '__main__':
    pytest.main([__file__])
