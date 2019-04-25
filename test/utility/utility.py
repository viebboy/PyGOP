# -*- coding: utf-8 -*-
"""
Utility for testing purpose


Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend as K
from GOP.utility import gop_operators

INPUT_DIM = 11
OUTPUT_DIM = 4


def get_generator(data):
    input_dim, output_dim, batch_size, steps = data
    x = np.random.rand(batch_size * steps, input_dim)
    y = np.random.rand(batch_size * steps, output_dim) if output_dim is not None else None

    def gen():
        while True:
            indices = list(range(batch_size * steps))
            random.shuffle(indices)

            for i in range(steps):
                start_idx = i * batch_size
                stop_idx = (i + 1) * batch_size
                if output_dim is not None:
                    yield x[indices[start_idx:stop_idx]], y[indices[start_idx:stop_idx]]
                else:
                    yield x[indices[start_idx:stop_idx]]

    return gen(), steps


def get_dummy_keras_model(input_dim, output_dim):
    inputs = Input((input_dim,))
    hiddens = Dense(10, activation='relu')(inputs)
    outputs = Dense(output_dim)(hiddens)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def mean_absolute_error_numpy(y_true, y_pred):
    return np.mean(np.abs(y_true.flatten() - y_pred.flatten()))


def mean_absolute_error_keras(y_true, y_pred):
    return K.mean(K.abs(K.flatten(y_true) - K.flatten(y_pred)))


def custom_nodal(x, w):
    return K.exp(-K.square(x * w))


def custom_pool(z):
    return K.min(z, axis=1)


def custom_activation(y):
    return K.relu(K.sigmoid(y))


def get_all_operators():
    nodal_set = gop_operators.get_default_nodal_set()
    pool_set = gop_operators.get_default_pool_set()
    activation_set = gop_operators.get_default_activation_set()

    all_operators = gop_operators.get_op_set(nodal_set, pool_set, activation_set)

    return all_operators


def get_random_model_data():
    all_operators = get_all_operators()
    no_op = len(all_operators)

    use_bias = random.choice([True, False])

    topology = [INPUT_DIM,
                [('gop', 20), (random.choice(['gop', 'mem']), 30)],
                [('gop', 40), (random.choice(['gop', 'mem']), 10)],
                (random.choice(['gop', 'dense']), OUTPUT_DIM)]

    op_sets = {'gop_0_0': all_operators[np.random.randint(0, no_op)],
               'gop_0_1': all_operators[np.random.randint(0, no_op)],
               'gop_1_0': all_operators[np.random.randint(0, no_op)],
               'gop_1_1': all_operators[np.random.randint(0, no_op)],
               'output': all_operators[np.random.randint(0, no_op)]}

    weights = {}
    if use_bias:
        # handle weights in 1st block in 1st and 2nd hidden layer
        weights['gop_0_0'] = [np.random.rand(1, INPUT_DIM, 20), np.zeros((20,))]
        weights['bn_0_0'] = [np.ones((20,)), np.zeros((20,)), np.zeros((20,)), np.ones((20,))]
        weights['gop_1_0'] = [np.random.rand(1, 50, 40), np.zeros((40,))]
        weights['bn_1_0'] = [np.ones((40,)), np.zeros((40,)), np.zeros((40,)), np.ones((40,))]

        # handle weights in 2nd block, 1st hidden layer
        if topology[1][1][0] == 'gop':
            weights['gop_0_1'] = [np.random.rand(1, INPUT_DIM, 30), np.zeros((30,))]
        else:
            weights['mem_pre_bn_0_1'] = [np.ones((INPUT_DIM,)), np.zeros(
                (INPUT_DIM,)), np.zeros((INPUT_DIM,)), np.ones((INPUT_DIM,))]
            weights['mem_0_1'] = [np.random.rand(INPUT_DIM, 30), np.zeros((30,))]

        weights['bn_0_1'] = [np.ones((30,)), np.zeros((30,)), np.zeros((30,)), np.ones((30,))]

        # handle weights in 2nd block, 2nd hidden layer
        if topology[2][1][0] == 'gop':
            weights['gop_1_1'] = [np.random.rand(1, 50, 10), np.zeros((10,))]
        else:
            weights['mem_pre_bn_1_1'] = [np.ones((50,)), np.zeros(
                (50,)), np.zeros((50,)), np.ones((50,))]
            weights['mem_1_1'] = [np.random.rand(50, 10), np.zeros((10,))]

        weights['bn_1_1'] = [np.ones((10,)), np.zeros((10,)), np.zeros((10,)), np.ones((10,))]

        # handle weights in output layer
        if topology[-1][0] == 'gop':
            weights['output'] = [np.random.rand(1, 50, OUTPUT_DIM), np.zeros((OUTPUT_DIM,))]
        else:
            weights['output'] = [np.random.rand(50, OUTPUT_DIM), np.zeros((OUTPUT_DIM,))]
    else:
        # handle weights in 1st block in 1st and 2nd hidden layer
        weights['gop_0_0'] = [np.random.rand(1, INPUT_DIM, 20)]
        weights['bn_0_0'] = [np.ones((20,)), np.zeros((20,)), np.zeros((20,)), np.ones((20,))]
        weights['gop_1_0'] = [np.random.rand(1, 50, 40)]
        weights['bn_1_0'] = [np.ones((40,)), np.zeros((40,)), np.zeros((40,)), np.ones((40,))]

        # handle weights in 2nd block, 1st hidden layer
        if topology[1][1][0] == 'gop':
            weights['gop_0_1'] = [np.random.rand(1, INPUT_DIM, 30)]
        else:
            weights['mem_pre_bn_0_1'] = [np.ones((INPUT_DIM,)), np.zeros(
                (INPUT_DIM,)), np.zeros((INPUT_DIM,)), np.ones((INPUT_DIM,))]
            weights['mem_0_1'] = [np.random.rand(INPUT_DIM, 30)]

        weights['bn_0_1'] = [np.ones((30,)), np.zeros((30,)), np.zeros((30,)), np.ones((30,))]

        # handle weights in 2nd block, 2nd hidden layer
        if topology[2][1][0] == 'gop':
            weights['gop_1_1'] = [np.random.rand(1, 50, 10)]
        else:
            weights['mem_pre_bn_1_1'] = [np.ones((50,)), np.zeros(
                (50,)), np.zeros((50,)), np.ones((50,))]
            weights['mem_1_1'] = [np.random.rand(50, 10)]

        weights['bn_1_1'] = [np.ones((10,)), np.zeros((10,)), np.zeros((10,)), np.ones((10,))]

        # handle weights in output layer
        if topology[-1][0] == 'gop':
            weights['output'] = [np.random.rand(1, 50, OUTPUT_DIM)]
        else:
            weights['output'] = [np.random.rand(50, OUTPUT_DIM)]

    model_data = {'topology': topology,
                  'op_sets': op_sets,
                  'weights': weights,
                  'output_activation': random.choice([None, 'softmax']),
                  'use_bias': use_bias,
                  'model': random.choice(['hemlgop',
                                          'hemlrn',
                                          'homlgop',
                                          'homlrn',
                                          'pop',
                                          'popfast',
                                          'popmemo',
                                          'popmemh'])}

    params = {'direction': 'lower',
              'input_dim': INPUT_DIM,
              'output_dim': OUTPUT_DIM,
              'input_dropout': random.choice([None, 0.2]),
              'dropout': random.choice([None, 0.2]),
              'dropout_finetune': random.choice([None, 0.2]),
              'weight_regularizer': random.choice([None, 1e-4]),
              'weight_regularizer_finetune': random.choice([None, 1e-4]),
              'weight_constraint': random.choice([None, 3.0]),
              'weight_constraint_finetune': random.choice([None, 3.0]),
              'optimizer': random.choice(['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']),
              'optimizer_parameters': random.choice([None, {'lr': 1e-2}]),
              'loss': random.choice(['mean_squared_error', mean_absolute_error_keras]),
              'metrics': ['mean_squared_error', random.choice(['mean_absolute_error', mean_absolute_error_keras])],
              'special_metrics': random.choice([None, [mean_absolute_error_numpy]]),
              'lr_train': (1e-3, 1e-4),
              'epoch_train': (1, 1),
              'lr_finetune': (1e-3, 1e-4),
              'epoch_finetune': (1, 1),
              'class_weight': None,
              'output_activation': model_data['output_activation'],
              'use_bias': use_bias}

    if params['loss'] not in params['metrics']:
        params['metrics'].append(params['loss'])

    if callable(params['loss']):
        params['convergence_measure'] = params['loss'].__name__
    else:
        params['convergence_measure'] = params['loss']

    return params, model_data


def get_random_popmem_states():
    params, _ = get_random_model_data()
    params['nodal_set'] = gop_operators.get_default_nodal_set()
    params['pool_set'] = gop_operators.get_default_pool_set()
    params['activation_set'] = gop_operators.get_default_activation_set()

    topology = [INPUT_DIM,
                [('gop', 20), ('mem', 30)],
                ('dense', OUTPUT_DIM)]

    weights = {}

    if params['use_bias']:
        weights['gop_0_0'] = [np.random.rand(1, INPUT_DIM, 20), np.zeros((20,))]
        weights['bn_0_0'] = [np.ones((20,)), np.zeros((20,)), np.zeros((20,)), np.ones((20,))]
        weights['mem_pre_bn_0_1'] = [np.ones((INPUT_DIM,)), np.zeros(
            (INPUT_DIM,)), np.zeros((INPUT_DIM,)), np.ones((INPUT_DIM,))]
        weights['mem_0_1'] = [np.random.rand(INPUT_DIM, 30), np.zeros((30,))]
        weights['bn_0_1'] = [np.ones((30,)), np.zeros((30,)), np.zeros((30,)), np.ones((30,))]
        weights['output'] = [np.random.rand(50, OUTPUT_DIM), np.zeros((OUTPUT_DIM,))]
    else:
        weights['gop_0_0'] = [np.random.rand(1, INPUT_DIM, 20), ]
        weights['bn_0_0'] = [np.ones((20,)), np.zeros((20,)), np.zeros((20,)), np.ones((20,))]
        weights['mem_pre_bn_0_1'] = [np.ones((INPUT_DIM,)), np.zeros(
            (INPUT_DIM,)), np.zeros((INPUT_DIM,)), np.ones((INPUT_DIM,))]
        weights['mem_0_1'] = [np.random.rand(INPUT_DIM, 30), ]
        weights['bn_0_1'] = [np.ones((30,)), np.zeros((30,)), np.zeros((30,)), np.ones((30,))]
        weights['output'] = [np.random.rand(50, OUTPUT_DIM), ]

    no_op = len(params['nodal_set']) * len(params['pool_set']) * len(params['activation_set'])

    op_set_indices = {'gop_0_0': random.choice(list(range(no_op)))}

    train_states = {'topology': topology,
                    'weights': weights,
                    'layer_iter': 1,
                    'op_set_indices': op_set_indices}

    params['direct_computation'] = random.choice([True, False])
    params['memory_regularizer'] = 1e-1
    p = np.random.rand(1)[0]
    params['min_energy_percentage'] = p if p > 0.5 else 1 - p

    return params, train_states


def get_random_states():
    params, _ = get_random_model_data()
    params['nodal_set'] = gop_operators.get_default_nodal_set()[:2]
    params['pool_set'] = gop_operators.get_default_pool_set()[:2]
    params['activation_set'] = gop_operators.get_default_activation_set()[:1]
    params['direct_computation'] = random.choice([True, False])
    params['least_square_regularizer'] = 1e-1
    params['block_size'] = 20

    topology = [INPUT_DIM,
                [('gop', 20), ],
                ('dense', OUTPUT_DIM)]

    weights = {}

    if params['use_bias']:
        weights['gop_0_0'] = [np.random.rand(1, INPUT_DIM, 20), np.zeros((20,))]
        weights['bn_0_0'] = [np.ones((20,)), np.zeros((20,)), np.zeros((20,)), np.ones((20,))]
        weights['output'] = [np.random.rand(20, OUTPUT_DIM), np.zeros((OUTPUT_DIM,))]
    else:
        weights['gop_0_0'] = [np.random.rand(1, INPUT_DIM, 20), ]
        weights['bn_0_0'] = [np.ones((20,)), np.zeros((20,)), np.zeros((20,)), np.ones((20,))]
        weights['output'] = [np.random.rand(20, OUTPUT_DIM), ]

    no_op = len(params['nodal_set']) * len(params['pool_set']) * len(params['activation_set'])

    op_set_indices = {'gop_0_0': random.choice(list(range(no_op)))}

    train_states = {'topology': topology,
                    'weights': weights,
                    'layer_iter': 0,
                    'block_iter': 1,
                    'op_set_indices': op_set_indices}

    return params, train_states
