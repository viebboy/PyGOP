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
import utility
import numpy as np
import copy
import random
from GOP.utility import misc
from keras import backend as K
from keras import optimizers as keras_optimizers
from GOP import models as gop_models

INPUT_DIM = utility.INPUT_DIM
OUTPUT_DIM = utility.OUTPUT_DIM
BATCH_SIZE = 32
STEPS = 4


def test_BestModelLogging():

    train_func = utility.get_generator
    train_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS]
    val_func = utility.get_generator
    val_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS]
    test_func = utility.get_generator
    test_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS]

    callback = misc.BestModelLogging(convergence_measure='train_mean_squared_error',
                                     direction='lower',
                                     special_metrics=None,
                                     train_func=train_func,
                                     train_data=train_data,
                                     val_func=val_func,
                                     val_data=val_data,
                                     test_func=test_func,
                                     test_data=test_data)

    model = utility.get_dummy_keras_model(INPUT_DIM, OUTPUT_DIM)
    model.compile('adam', 'mse', ['mse', ])

    train_gen, train_steps = train_func(train_data)
    val_gen, val_steps = val_func(val_data)

    model.fit_generator(train_gen,
                        train_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=1,
                        callbacks=[callback, ],
                        verbose=0)

    assert callback.measure < np.inf
    assert len(callback.performance['train_mean_squared_error']) == 1
    assert len(callback.performance['val_mean_squared_error']) == 1
    assert len(callback.performance['test_mean_squared_error']) == 1


def test_get_he_init():
    initializer = misc.get_he_init(10, 20)
    data = initializer((4, 5))
    assert K.int_shape(data) == (4, 5)


def test_initialize_states(tmpdir):
    tmp_dir = tmpdir.dirname
    model_name = 'test_model'
    if os.path.exists(os.path.join(tmpdir.dirname, model_name)):
        shutil.rmtree(os.path.join(tmpdir.dirname, model_name))

    os.mkdir(os.path.join(tmpdir.dirname, model_name))

    parameters = {'tmp_dir': tmp_dir,
                  'model_name': model_name,
                  'use_bias': True,
                  'output_activation': None,
                  'input_dim': INPUT_DIM,
                  'output_dim': OUTPUT_DIM}

    train_states = misc.initialize_states(parameters, 'hemlgop')

    train_states_file = os.path.join(tmp_dir, model_name, 'train_states.pickle')

    with open(train_states_file, 'wb') as fid:
        pickle.dump(train_states, fid)

    assert train_states == misc.initialize_states(parameters, 'hemlgop')

    shutil.rmtree(os.path.join(tmpdir.dirname, model_name))

    with pytest.raises(OSError):
        misc.initialize_states(parameters, 'hemlgop')


def test_pickle_generator(tmpdir):
    train_func = utility.get_generator
    train_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS]
    val_func = utility.get_generator
    val_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS]
    test_func = utility.get_generator
    test_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS]

    filename = os.path.join(tmpdir.dirname, 'data_generator.pickle')

    # dump data
    misc.pickle_generator(filename, train_func, train_data,
                          val_func, val_data, test_func, test_data)

    assert os.path.exists(filename)
    with open(filename, 'rb') as fid:
        data = pickle.load(fid)

    assert data['train_data'] == train_data
    assert data['train_func'] == dill.dumps(train_func, recurse=True)

    # remove data
    os.remove(filename)

    # dump data
    misc.pickle_generator(filename, train_func, train_data, None, None, test_func, test_data)

    assert os.path.exists(filename)
    with open(filename, 'rb') as fid:
        data = pickle.load(fid)

    assert data['val_data'] is None
    os.remove(filename)


def test_unpickle_generator(tmpdir):
    train_func = utility.get_generator
    train_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS]
    filename = os.path.join(tmpdir.dirname, 'data_generator.pickle')

    # dump data
    misc.pickle_generator(filename, train_func, train_data, None, None, None, None)

    train_func_recovered, train_data_recovered, val_func, val_data, test_func, test_data = misc.unpickle_generator(
        filename)

    assert val_data is None
    assert val_func is None
    assert test_data is None
    assert test_func is None
    assert train_data == train_data_recovered

    train_gen, train_steps = train_func(train_data)
    assert train_steps == STEPS

    for i in range(3):
        next(train_gen)

    os.remove(filename)


def test_get_op_set_index():
    no_op_set = 100
    no_machine = 1
    machine_no = 0

    start_idx, stop_idx = misc.get_op_set_index(no_op_set, no_machine, machine_no)
    assert start_idx == 0
    assert stop_idx == 100

    no_op_set = 100
    no_machine = 2
    machine_no = 1

    start_idx, stop_idx = misc.get_op_set_index(no_op_set, no_machine, machine_no)
    assert start_idx == 50
    assert stop_idx == 100

    no_op_set = 100
    no_machine = 3
    machine_no = 1

    start_idx, stop_idx = misc.get_op_set_index(no_op_set, no_machine, machine_no)
    assert start_idx == 34
    assert stop_idx == 68


def test_evaluate():

    model = utility.get_dummy_keras_model(INPUT_DIM, OUTPUT_DIM)
    model.compile('adam', 'mse', ['mse', ])

    gen, steps = utility.get_generator([INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    performance = misc.evaluate(model, gen, steps, None, None, None, None)

    assert performance['val_mean_squared_error'] is None
    assert performance['test_mean_squared_error'] is None


def test_evaluate_special_metrics():

    model = utility.get_dummy_keras_model(INPUT_DIM, OUTPUT_DIM)
    model.compile('adam', 'mse', ['mse', ])

    gen, steps = utility.get_generator([INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS])

    performance = misc.evaluate_special_metrics(
        model, [utility.mean_absolute_error_numpy], gen, steps)

    assert len(performance) == 1


def test_check_convergence():
    new_measure = 0.1
    old_measure = 0.1
    direction = 'higher'
    threshold = 1e-4

    assert misc.check_convergence(new_measure, old_measure, direction, threshold)

    new_measure = 0.101
    old_measure = 0.1
    direction = 'higher'
    threshold = 1e-4

    assert misc.check_convergence(new_measure, old_measure, direction, threshold) is False

    new_measure = 0.101
    old_measure = 0.1
    direction = 'lower'
    threshold = 1e-4

    assert misc.check_convergence(new_measure, old_measure, direction, threshold)

    new_measure = 0.09
    old_measure = 0.1
    direction = 'lower'
    threshold = 1e-4

    assert misc.check_convergence(new_measure, old_measure, direction, threshold) is False


def test_pickle_custom_metrics(tmpdir):
    filename = os.path.join(tmpdir.dirname, 'custom_metrics.pickle')
    has_custom_metric = misc.pickle_custom_metrics(
        ['mse', utility.mean_absolute_error_keras], filename)

    assert has_custom_metric
    assert os.path.exists(filename)

    with open(filename, 'rb') as fid:
        metrics = pickle.load(fid)

    assert metrics['names'] == ['mse', 0]

    os.remove(filename)

    has_custom_metric = misc.pickle_custom_metrics(['mse', 'categorical_crossentropy'], filename)
    assert has_custom_metric is False


def test_unpickle_custom_metrics(tmpdir):
    filename = os.path.join(tmpdir.dirname, 'custom_metrics.pickle')
    misc.pickle_custom_metrics(['mse', utility.mean_absolute_error_keras], filename)

    metrics = misc.unpickle_custom_metrics(filename)

    for metric in metrics:
        if isinstance(metric, str):
            assert metric == 'mse'
        if callable(metric):
            assert metric.__name__ == 'mean_absolute_error_keras'

    y_true = K.ones((10, 20))
    y_pred = K.zeros((10, 20))

    assert K.eval(utility.mean_absolute_error_keras(
        y_true, y_pred)) == K.eval(metrics[1](y_true, y_pred))

    os.remove(filename)


def test_pickle_custom_loss(tmpdir):

    filename = os.path.join(tmpdir.dirname, 'custom_loss.pickle')
    has_custom_loss = misc.pickle_custom_loss(utility.mean_absolute_error_keras, filename)

    assert has_custom_loss
    assert os.path.exists(filename)

    with open(filename, 'rb') as fid:
        loss = pickle.load(fid)

    assert loss['function_string'] == dill.dumps(utility.mean_absolute_error_keras)

    os.remove(filename)

    has_custom_loss = misc.pickle_custom_loss('categorical_crossentropy', filename)
    assert has_custom_loss is False


def test_unpickle_custom_loss(tmpdir):

    filename = os.path.join(tmpdir.dirname, 'custom_loss.pickle')
    misc.pickle_custom_loss(utility.mean_absolute_error_keras, filename)

    loss = misc.unpickle_custom_loss(filename)

    y_true = K.ones((10, 20))
    y_pred = K.zeros((10, 20))

    assert K.eval(utility.mean_absolute_error_keras(y_true, y_pred)) == K.eval(loss(y_true, y_pred))

    assert loss.__name__ == 'mean_absolute_error_keras'

    os.remove(filename)


def test_pickle_special_metrics(tmpdir):

    filename = os.path.join(tmpdir.dirname, 'special_metrics.pickle')
    has_special_metric = misc.pickle_special_metrics([utility.mean_absolute_error_numpy], filename)

    assert has_special_metric
    assert os.path.exists(filename)

    os.remove(filename)


def test_unpickle_special_metrics(tmpdir):

    filename = os.path.join(tmpdir.dirname, 'special_metrics.pickle')
    misc.pickle_special_metrics([utility.mean_absolute_error_numpy], filename)

    metrics = misc.unpickle_special_metrics(filename)

    assert metrics[0].__name__ == 'mean_absolute_error_numpy'

    y_true = np.random.rand(10, 20)
    y_pred = np.random.rand(10, 20)

    assert utility.mean_absolute_error_numpy(y_true, y_pred) == metrics[0](y_true, y_pred)

    os.remove(filename)


def test_pickle_custom_operators(tmpdir):

    filename = os.path.join(tmpdir.dirname, 'custom_operators.pickle')
    nodal_set = ['multiplication', utility.custom_nodal]
    pool_set = ['summation', utility.custom_pool]
    activation_set = ['relu', utility.custom_activation]

    has_custom_operator = misc.pickle_custom_operators(
        nodal_set, pool_set, activation_set, filename)
    assert has_custom_operator
    assert os.path.exists(filename)

    os.remove(filename)

    nodal_set = ['multiplication', ]
    pool_set = ['summation', ]
    activation_set = ['relu', ]

    has_custom_operator = misc.pickle_custom_operators(
        nodal_set, pool_set, activation_set, filename)
    assert has_custom_operator is False


def test_unpickle_custom_operators(tmpdir):

    filename = os.path.join(tmpdir.dirname, 'custom_operators.pickle')
    nodal_set = ['multiplication', utility.custom_nodal]
    pool_set = ['summation', utility.custom_pool]
    activation_set = ['relu', utility.custom_activation]

    misc.pickle_custom_operators(nodal_set, pool_set, activation_set, filename)
    nodal_set_recovered, pool_set_recovered, activation_set_recovered = misc.unpickle_custom_operators(
        filename)
    assert nodal_set_recovered[1].__name__ == 'custom_nodal'
    assert pool_set_recovered[1].__name__ == 'custom_pool'
    assert activation_set_recovered[1].__name__ == 'custom_activation'

    x = K.variable(np.random.rand(10, 20))
    w = K.variable(np.random.rand(10, 20))
    z = K.variable(np.random.rand(10, 20))
    y = K.variable(np.random.rand(10, 20))

    assert np.allclose(K.eval(utility.custom_nodal(x, w)), K.eval(nodal_set_recovered[1](x, w)))
    assert np.allclose(K.eval(utility.custom_pool(z)), K.eval(pool_set_recovered[1](z)))
    assert np.allclose(K.eval(utility.custom_activation(y)), K.eval(activation_set_recovered[1](y)))


def test_get_gpu_str():

    devices = [0, 2, 3]
    assert misc.get_gpu_str(devices) == '0,2,3'


def test_partition_indices():

    start_idx = 0
    stop_idx = 100
    no_partition = 3
    start_indices, stop_indices = misc.partition_indices(start_idx, stop_idx, no_partition)

    assert start_indices[0] == 0
    assert stop_indices[0] == 34
    assert start_indices[1] == 34
    assert stop_indices[1] == 68
    assert start_indices[2] == 68
    assert stop_indices[2] == 100

    start_idx = 33
    stop_idx = 99
    no_partition = 2
    start_indices, stop_indices = misc.partition_indices(start_idx, stop_idx, no_partition)

    assert start_indices[0] == 33
    assert stop_indices[0] == 66
    assert start_indices[1] == 66
    assert stop_indices[1] == 99


def test_remove_files(tmpdir):

    files = []
    for i in range(10):
        f = os.path.join(tmpdir.dirname, '%d.txt' % i)
        open(f, 'w').close()
        files.append(f)

    misc.remove_files(files)
    for f in files:
        assert not os.path.exists(f)

    with pytest.raises(OSError):
        misc.remove_files(files)


def test_dump_data(tmpdir):

    tmp_dir = tmpdir.dirname
    model_name = 'test_model'
    if os.path.exists(os.path.join(tmp_dir, model_name)):
        shutil.rmtree(os.path.join(tmp_dir, model_name))
    os.mkdir(os.path.join(tmp_dir, model_name))

    params = {'tmp_dir': tmp_dir,
              'model_name': model_name,
              'loss': 'mse',
              'metrics': ['mse', utility.mean_absolute_error_keras],
              'special_metrics': [utility.mean_absolute_error_numpy],
              'nodal_set': ['multiplication', utility.custom_nodal],
              'pool_set': ['summation'],
              'activation_set': ['relu']}

    train_states = {'name': 'state1'}

    train_func = utility.get_generator
    train_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS]
    val_func = None
    val_data = None
    test_func = utility.get_generator
    test_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS]

    misc.dump_data(params, train_states, train_func, train_data,
                   val_func, val_data, test_func, test_data)

    assert os.path.exists(os.path.join(tmp_dir, model_name, 'params.pickle'))
    assert os.path.exists(os.path.join(tmp_dir, model_name, 'train_states_tmp.pickle'))
    assert os.path.exists(os.path.join(tmp_dir, model_name, 'data.pickle'))
    assert not os.path.exists(os.path.join(tmp_dir, model_name, 'custom_loss.pickle'))
    assert os.path.exists(os.path.join(tmp_dir, model_name, 'custom_metrics.pickle'))
    assert os.path.exists(os.path.join(tmp_dir, model_name, 'special_metrics.pickle'))
    assert os.path.exists(os.path.join(tmp_dir, model_name, 'custom_operators.pickle'))

    with open(os.path.join(tmp_dir, model_name, 'train_states_tmp.pickle'), 'rb') as fid:
        train_states_recovered = pickle.load(fid)

    assert train_states_recovered['name'] == 'state1'

    with open(os.path.join(tmp_dir, model_name, 'params.pickle'), 'rb') as fid:
        params_recovered = pickle.load(fid)

    assert params_recovered['metrics'] == []
    assert params_recovered['special_metrics'] == []
    assert params_recovered['nodal_set'] == []
    assert params_recovered['pool_set'] == []
    assert params_recovered['activation_set'] == []

    train_states['name'] = 'state2'

    misc.dump_data(params, train_states, train_func, train_data,
                   val_func, val_data, test_func, test_data)

    with open(os.path.join(tmp_dir, model_name, 'train_states_tmp.pickle'), 'rb') as fid:
        train_states_recovered = pickle.load(fid)

    assert train_states_recovered['name'] == 'state2'


def test_reconstruct_parameters(tmpdir):

    tmp_dir = tmpdir.dirname
    model_name = 'test_model'
    if os.path.exists(os.path.join(tmp_dir, model_name)):
        shutil.rmtree(os.path.join(tmp_dir, model_name))
    os.mkdir(os.path.join(tmp_dir, model_name))

    params = {'tmp_dir': tmp_dir,
              'model_name': model_name,
              'loss': 'mse',
              'metrics': ['mse', utility.mean_absolute_error_keras],
              'special_metrics': [utility.mean_absolute_error_numpy],
              'nodal_set': ['multiplication', utility.custom_nodal],
              'pool_set': ['summation'],
              'activation_set': ['relu']}

    train_states = {'name': 'state1'}

    train_func = utility.get_generator
    train_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS]
    val_func = None
    val_data = None
    test_func = utility.get_generator
    test_data = [INPUT_DIM, OUTPUT_DIM, BATCH_SIZE, STEPS]

    misc.dump_data(params, train_states, train_func, train_data,
                   val_func, val_data, test_func, test_data)

    params_reconstruct = copy.deepcopy(params)
    params_reconstruct['metrics'] = []
    params_reconstruct['special_metrics'] = []
    params_reconstruct['nodal_set'] = []
    params_reconstruct['pool_set'] = []
    params_reconstruct['activation_set'] = []

    params_reconstruct = misc.reconstruct_parameters(params_reconstruct)
    assert params_reconstruct == params

    shutil.rmtree(os.path.join(tmp_dir, model_name))


def test_get_optimizer_instance():

    optimizer = random.choice(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'])
    parameters = random.choice([None, {'lr': 1.0}])
    optimizer_obj = misc.get_optimizer_instance(optimizer, parameters)

    assert isinstance(optimizer_obj, keras_optimizers.Optimizer)
    if parameters is not None:
        assert K.eval(optimizer_obj.lr) == 1.0


def test_map_operator_from_index():

    nodal_set = ['multiplication', 'exp']
    pool_set = ['summation', utility.custom_pool]
    activation_set = ['relu']
    total_op_sets = [(nodal, pool, activation)
                     for nodal in nodal_set for pool in pool_set for activation in activation_set]

    op_set_indices = {'gop_%d' % k: k for k in range(len(total_op_sets))}

    op_sets = misc.map_operator_from_index(op_set_indices, nodal_set, pool_set, activation_set)

    layer_names = ['gop_' + str(k) for k in range(len(total_op_sets))]
    keys = list(op_sets.keys())
    keys.sort()
    assert keys == layer_names

    for k in range(len(total_op_sets)):
        assert op_sets['gop_' + str(k)] == total_op_sets[k]


def test_check_model_parameters(tmpdir):

    model = gop_models.HeMLGOP()
    default_params = model.get_default_parameters()
    params = copy.deepcopy(default_params)

    with pytest.raises(AssertionError, match='This model requires a read/writeable temporary directory'):
        misc.check_model_parameters(params, default_params)

    params['tmp_dir'] = tmpdir.dirname
    with pytest.raises(AssertionError, match='This model requires a unique name in temporary directory'):
        misc.check_model_parameters(params, default_params)

    params['model_name'] = 'test_HeMLGOP'
    params['tmp_dir'] = 'this is a random dir'

    with pytest.raises(AssertionError, match='does not exist'):
        misc.check_model_parameters(params, default_params)

    params['tmp_dir'] = tmpdir.dirname
    with pytest.raises(AssertionError, match='Input dimension'):
        misc.check_model_parameters(params, default_params)

    params['input_dim'] = utility.INPUT_DIM
    with pytest.raises(AssertionError, match='Output dimension'):
        misc.check_model_parameters(params, default_params)

    params['output_dim'] = utility.OUTPUT_DIM
    assert os.path.exists(os.path.join(tmpdir.dirname, 'test_HeMLGOP'))

    params['cluster'] = True
    with pytest.raises(AssertionError, match='batchjob_parameters'):
        misc.check_model_parameters(params, default_params)

    cluster_params = ['name', 'mem', 'core', 'partition', 'time', 'no_machine', 'python_cmd']
    job_config = {name: None for name in cluster_params}
    params['batchjob_parameters'] = job_config

    for name in cluster_params:
        del params['batchjob_parameters'][name]
        with pytest.raises(AssertionError, match=name):
            misc.check_model_parameters(params, default_params)
        params['batchjob_parameters'][name] = None
    params['cluster'] = False

    params['search_computation'] = 'cpu'
    with pytest.raises(AssertionError, match='search_computation'):
        misc.check_model_parameters(params, default_params)

    params['search_computation'] = ('cpu', 'device0')
    with pytest.raises(AssertionError, match='"search_computation"="cpu"'):
        misc.check_model_parameters(params, default_params)

    params['search_computation'] = ('gpu', 0)
    with pytest.raises(AssertionError, match='"search_computation"="gpu"'):
        misc.check_model_parameters(params, default_params)
    params['search_computation'] = ('cpu', 1)

    params['finetune_computation'] = ('gpu', 0)
    with pytest.raises(AssertionError, match='"finetune_computation"="gpu"'):
        misc.check_model_parameters(params, default_params)
    params['finetune_computation'] = ('cpu', 1)

    params['metrics'] = utility.mean_absolute_error_keras
    with pytest.raises(AssertionError, match='metrics should be given as a single/list/tuple'):
        misc.check_model_parameters(params, default_params)
    params['metrics'] = default_params['metrics']

    params['special_metrics'] = utility.mean_absolute_error_keras
    with pytest.raises(AssertionError, match='special_metrics should be given as a list/tuple'):
        misc.check_model_parameters(params, default_params)
    params['special_metrics'] = [None]
    with pytest.raises(AssertionError, match='special_metrics should be a list/tuple of callable'):
        misc.check_model_parameters(params, default_params)
    params['special_metrics'] = None

    params['convergence_measure'] = random.choice([None, 3])
    with pytest.raises(AssertionError, match='convergence_measure'):
        misc.check_model_parameters(params, default_params)

    params['convergence_measure'] = 'CE'
    params['metrics'] = ['mse', utility.mean_absolute_error_keras]
    params['special_metrics'] = [utility.mean_absolute_error_numpy]
    with pytest.raises(AssertionError, match='should belong to the list of metrics'):
        misc.check_model_parameters(params, default_params)
    params['convergence_measure'] = 'mean_absolute_error_keras'

    params['loss'] = None
    with pytest.raises(AssertionError, match='loss should be'):
        misc.check_model_parameters(params, default_params)
    params['loss'] = default_params['loss']

    params['direction'] = 'asd'
    with pytest.raises(AssertionError, match='direction'):
        misc.check_model_parameters(params, default_params)
    params['direction'] = default_params['direction']

    params['optimizer'] = keras_optimizers.Adam()
    with pytest.raises(AssertionError, match='Optimizer should be a str'):
        misc.check_model_parameters(params, default_params)
    params['optimizer'] = 'SGD'
    with pytest.raises(AssertionError, match='Only support Keras optimizers'):
        misc.check_model_parameters(params, default_params)
    params['optimizer'] = 'sgd'

    params['optimizer_parameters'] = 3
    with pytest.raises(AssertionError, match='"optimizer_parameters" should be'):
        misc.check_model_parameters(params, default_params)


if __name__ == '__main__':
    pytest.main([__file__])
