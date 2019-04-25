#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import pytest
import numpy as np
from GOP.utility import gop_operators
from keras import backend as K


def test_multiplication_():
    x_np = np.random.rand(1, 10, 20)
    w_np = np.random.rand(1, 10, 20)

    x = K.variable(x_np)
    w = K.variable(w_np)

    assert np.allclose(x_np * w_np, K.eval(gop_operators.multiplication_(x, w)))


def test_exponential_():
    x_np = np.random.rand(1, 10, 20)
    w_np = np.random.rand(1, 10, 20)

    x = K.variable(x_np)
    w = K.variable(w_np)

    assert np.allclose(np.exp(x_np * w_np) - 1.0, K.eval(gop_operators.exponential_(x, w)))


def test_harmonic_():
    x_np = np.random.rand(1, 10, 20)
    w_np = np.random.rand(1, 10, 20)

    x = K.variable(x_np)
    w = K.variable(w_np)

    assert np.allclose(np.sin(x_np * w_np), K.eval(gop_operators.harmonic_(x, w)))


def test_quadratic_():
    x_np = np.random.rand(1, 10, 20)
    w_np = np.random.rand(1, 10, 20)

    x = K.variable(x_np)
    w = K.variable(w_np)

    assert np.allclose(np.square(x_np * w_np), K.eval(gop_operators.quadratic_(x, w)))


def test_gaussian_():
    x_np = np.random.rand(1, 10, 20)
    w_np = np.random.rand(1, 10, 20)

    x = K.variable(x_np)
    w = K.variable(w_np)

    assert np.allclose(w_np * np.exp(-np.square(x_np) * w_np), K.eval(gop_operators.gaussian_(x, w)))


def test_dog_():
    x_np = np.random.rand(1, 10, 20)
    w_np = np.random.rand(1, 10, 20)

    x = K.variable(x_np)
    w = K.variable(w_np)

    assert np.allclose(w_np * x_np * np.exp(-np.square(x_np) * w_np), K.eval(gop_operators.dog_(x, w)))


def test_sum_():
    x_np = np.random.rand(1, 10, 20)
    x = K.variable(x_np)

    assert np.allclose(np.sum(x_np, -2), K.eval(gop_operators.sum_(x)))


def test_correlation1_():
    x_np = np.random.rand(1, 10, 20)
    y_np = np.pad(x_np[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')

    x = K.variable(x_np)

    assert np.allclose(np.sum(x_np * y_np, -2), K.eval(gop_operators.correlation1_(x)))


def test_correlation2_():
    x_np = np.random.rand(1, 10, 20)
    y_np = np.pad(x_np[:, 1:, :], ((0, 0), (0, 1), (0, 0)), 'constant')
    z_np = np.pad(x_np[:, 2:, :], ((0, 0), (0, 2), (0, 0)), 'constant')
    x = K.variable(x_np)

    assert np.allclose(np.sum(x_np * y_np * z_np, -2), K.eval(gop_operators.correlation2_(x)))


def test_maximum_():
    x_np = np.random.rand(1, 10, 20)
    x = K.variable(x_np)

    assert np.allclose(np.max(x_np, -2), K.eval(gop_operators.maximum_(x)))


def test_sigmoid_():
    x_np = np.random.rand(1, 10, 20)
    x = K.variable(x_np)

    assert np.allclose(1.0 / (1.0 + np.exp(-x_np)), K.eval(gop_operators.sigmoid_(x)))


def test_tanh_():
    x_np = np.random.rand(1, 10, 20)
    x = K.variable(x_np)

    assert np.allclose(np.tanh(x_np), K.eval(gop_operators.tanh_(x)))


def test_relu_():
    x_np = np.random.rand(1, 10, 20)
    x = K.variable(x_np)

    assert np.allclose((x_np > 0) * x_np, K.eval(gop_operators.relu_(x)))


def test_soft_linear_():
    x_np = np.random.rand(1, 10, 20)
    x = K.variable(x_np)

    assert np.allclose(np.log(1.0 + np.exp(-x_np)), K.eval(gop_operators.soft_linear_(x)))


def test_inverse_absolute_():
    x_np = np.random.rand(1, 10, 20)
    x = K.variable(x_np)

    assert np.allclose(x_np / (1.0 + np.abs(x_np)), K.eval(gop_operators.inverse_absolute_(x)))


def test_exp_linear_():
    x_np = np.random.rand(1, 10, 20)
    x = K.variable(x_np)

    assert np.allclose(np.where(x_np < 0, np.exp(x_np) - 1, x_np),
                       K.eval(gop_operators.exp_linear_(x)))


def test_get_default_nodal_set():
    nodal_set = ['multiplication', 'exponential', 'harmonic', 'quadratic', 'gaussian', 'dog']
    assert nodal_set == gop_operators.get_default_nodal_set()


def test_get_default_pool_set():
    pool_set = ['sum', 'correlation1', 'correlation2', 'maximum']
    assert pool_set == gop_operators.get_default_pool_set()


def test_get_default_activation_set():

    activation_set = ['sigmoid', 'relu', 'tanh', 'soft_linear', 'inverse_absolute', 'exp_linear']
    assert activation_set == gop_operators.get_default_activation_set()


def test_get_op_set():
    nodal_set = ['a', 'b', 'c']
    pool_set = ['x', 'y', 'z']
    activation_set = ['m', 'n', 'p']

    op_sets = [(nodal, pool, activation)
               for nodal in nodal_set for pool in pool_set for activation in activation_set]

    op_set_recovered = gop_operators.get_op_set(nodal_set, pool_set, activation_set)

    assert tuple(op_sets) == op_set_recovered


def test_get_random_op_set():
    nodal_set = gop_operators.get_default_nodal_set()
    pool_set = gop_operators.get_default_pool_set()
    activation_set = gop_operators.get_default_activation_set()

    op_set = gop_operators.get_random_op_set()

    assert op_set[0] in nodal_set
    assert op_set[1] in pool_set
    assert op_set[2] in activation_set


def test_get_nodal_operator():
    nodal_set = gop_operators.get_default_nodal_set()
    for nodal in nodal_set:
        assert callable(gop_operators.get_nodal_operator(nodal))
        assert gop_operators.get_nodal_operator(nodal).__name__ == nodal + '_'


def test_get_pool_operator():
    pool_set = gop_operators.get_default_pool_set()
    for pool in pool_set:
        assert callable(gop_operators.get_pool_operator(pool))
        assert gop_operators.get_pool_operator(pool).__name__ == pool + '_'


def test_get_activation_operator():
    activation_set = gop_operators.get_default_activation_set()
    for activation in activation_set:
        assert callable(gop_operators.get_activation_operator(activation))
        assert gop_operators.get_activation_operator(activation).__name__ == activation + '_'


if __name__ == '__main__':
    pytest.main([__file__])
