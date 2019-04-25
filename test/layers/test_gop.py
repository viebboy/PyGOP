#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import pytest
import random
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from GOP.layers.gop import GOP
from GOP.utility import gop_operators

INPUT_DIM = 10
OUTPUT_DIM = 3


def create_gop_model():
    op_set = gop_operators.get_random_op_set()
    regularizer = random.choice([None, 1e-4])
    constraint = random.choice([None, 4.0])
    trainable = random.choice([True, False])

    inputs = Input((INPUT_DIM,))
    hiddens = GOP(units=3,
                  op_set=op_set,
                  regularizer=regularizer,
                  constraint=constraint,
                  trainable=trainable,
                  name='GOP')(inputs)

    outputs = Dense(OUTPUT_DIM)(hiddens)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='mse', metrics=['mse', ])

    return model, trainable


def test_GOP():
    for i in range(3):
        model, trainable = create_gop_model()
        x = np.random.rand(64, INPUT_DIM)
        y = np.random.rand(64, OUTPUT_DIM)
        model.fit(x, y)


if __name__ == '__main__':
    pytest.main([__file__])
