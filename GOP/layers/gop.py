#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
GOP Keras layer definition 


Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""
from keras import backend as K
from keras.engine.topology import Layer
from keras import constraints, regularizers
from ..utility import misc, gop_operators
import tensorflow as tf

class GOP(Layer):
    """Generalized Operational Perceptron
    
    Args:
        units (int): Number of neurons
        op_set (tuple): Operator set with format (nodal, pool, activation)
        regularizer (float): Weight decay coefficient, default None
        constraint (float): Max-norm constraint, default None
        use_bias (bool): Use bias if True
        trainable (bool): Allow weight updates if True
    
        
    """
    def __init__(self,
                 units,
                 op_set,
                 regularizer=None,
                 constraint=None, 
                 use_bias = True, 
                 trainable = True,
                 **kwargs):
        
        self.units = units
        self.nodal = gop_operators.get_nodal_operator(op_set[0])
        self.pool = gop_operators.get_pool_operator(op_set[1])
        self.activation = gop_operators.get_activation_operator(op_set[2])
        self.regularizer = regularizers.l2(regularizer) if regularizer is not None else None
        self.constraint = constraints.max_norm(constraint, axis=1) if constraint is not None else None
        self.use_bias = use_bias
        self.trainable = trainable
        
        super(GOP, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.W = self.add_weight(name='W', 
                                      shape=(1, input_shape[-1], self.units),
                                      initializer=misc.get_he_init(input_shape[-1], self.units),
                                      regularizer=self.regularizer,
                                      constraint=self.constraint,
                                      trainable=self.trainable)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', 
                                          shape=(self.units,),
                                          initializer='zeros',
                                          trainable=self.trainable)
            
        super(GOP, self).build(input_shape)  

    def call(self, x):
        x = tf.expand_dims(x,axis=-1)

        x = self.nodal(x, self.W)
        x = self.pool(x)
        
        if self.use_bias:
            x=K.bias_add(x,self.bias)
        
        x=self.activation(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    