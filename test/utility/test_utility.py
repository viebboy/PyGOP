#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import os, sys
import numpy as np
import inspect
import tensorflow as tf

cwd = os.getcwd()

path = os.path.dirname(os.path.dirname(cwd))

if not path in sys.path:
    sys.path.append(path)
    
from GOP.utility import gop_operators, gop_utils

def scale(x):
    return x*0.01


def special_metric(y_true, y_pred):
    return np.mean(np.abs(y_true.flatten() - y_pred.flatten()))

def get_hemlgop_model(input_dim, output_dim):
    topology = [input_dim, [('gop',20)], ('dense',output_dim)]
    op_sets = {'gop_0_0': gop_operators.get_random_op_set()}
    
    model = gop_utils.network_builder(topology, op_sets)
    
    return model

def tf_mae(y_true,y_pred):
    return tf.reduce_sum(tf.abs(tf.reshape(y_true, (-1,)) - tf.reshape(y_pred, (-1,))))
    
def acc(y_true, y_pred):
    return np.sum(np.argmax(y_true, axis=-1)== np.argmax(y_pred, axis=-1))/ float(y_true.shape[0])
