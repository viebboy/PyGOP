#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Definition of GOP operators and related utilities


Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import tensorflow as tf
global _FLOATX
_FLOATX = str('float64')

def multiplication_(x, w):
    return x*w

def exponential_(x, w):
    y = x*w
    y = tf.clip_by_value(y, -10.0, 10.0)
    y = tf.exp(y) - 1.0    
    return y

def harmonic_(x, w):
    return tf.sin(x*w)

def quadratic_(x, w):
    return tf.square(tf.clip_by_value(x, -10.0, 10.0)*w)

def gaussian_(x, w):
    y = tf.square(x)*w
    y = tf.clip_by_value(y, -10.0, 10.0)
    return w*tf.exp(-y)

def dog_(x, w):
    y = tf.clip_by_value(tf.square(tf.clip_by_value(x, -10.0, 10.0))*w, -10.0, 10.0)
    return w*x*tf.exp(-y)

def sum_(x):
    return tf.reduce_sum(x, axis=-2)

def corr1_(x):
    y = tf.pad(x[:,1:,:], paddings=((0,0), (0,1), (0,0)))
    return tf.reduce_sum(x*y, axis=1)

def corr2_(x):
    y = tf.pad(x[:,1:,:], paddings=((0,0), (0,1), (0,0)))
    z = tf.pad(x[:,2:,:], paddings=((0,0), (0,2), (0,0)))
    return tf.reduce_sum(x*y*z, axis=1)

def max_(x):
    return tf.reduce_max(x, axis=-2)

def sigmoid_(x):
    return tf.sigmoid(x)

def tanh_(x):
    return tf.tanh(x)

def relu_(x):
    return tf.nn.relu(x)

def soft_linear_(x):
    y = tf.clip_by_value(x, -10.0, 10.0)
    return tf.log(1.0 + tf.exp(-y))

def inverse_absolute_(x):
    return x / (1.0 + tf.abs(x))

def exp_linear_(x):
    return tf.nn.elu(x)


    
def get_default_nodal_set():
    """Return the names of supported nodal operators
    
    Returns:
        a tuple of nodal operator names
    """
    output = ['multiplication','exponential','harmonic','quadratic','gaussian','dog']
    #output = ('harmonic','multiplication')
    
    return output

def get_default_pool_set():
    """Return the names of supported pooling operators
    
    Returns:
        a tuple of pooling operator names
    """
    output = ['sum','correlation1','correlation2','maximum']
    #output = ('sum', 'correlation1', 'correlation2','maximum')
    
    return output

def get_default_activation_set():
    """Return the names of supported activation operators
    
    Returns:
        a tuple of activation operator names
    """
    output = ['sigmoid','relu','tanh','soft_linear', 'inverse_absolute', 'exp_linear']
    #output = ('sigmoid','relu', 'tanh', 'inverse_absolute')
    
    return output

def get_op_set(nodal_set, pool_set, activation_set):
    """Create tuple of operator set (in tuple)
    
    """
    op_set = [(nodal, pool, act) for nodal in nodal_set for pool in pool_set for act in activation_set]
    
    return tuple(op_set)

def get_random_op_set():
    """Get a random operator set
    
    """
    import random
    return (random.choice(get_default_nodal_set()), 
            random.choice(get_default_pool_set()), 
            random.choice(get_default_activation_set()))

def get_nodal_operator(instance):
    """Get nodal operator in a callable
    
    """
    if isinstance(instance, str):
        supported_operator = get_default_nodal_set()
        if instance in supported_operator:
            if instance == 'multiplication':
                return multiplication_
            elif instance == 'exponential':
                return exponential_
            elif instance == 'harmonic':
                return harmonic_
            elif instance == 'quadratic':
                return quadratic_
            elif instance == 'gaussian':
                return gaussian_
            elif instance == 'dog':
                return dog_
            else:
                raise "Nodal names mismatch, given ´´%s´´, but only supported ´´%s´´" %(instance, supported_operator)
        else:
            raise "Nodal operator ´´%s´´ not supported!"%instance
    
    else:
        assert callable(instance), "The given nodal operator is neither in default list nor callable"
        return instance
    
def get_pool_operator(instance):
    """Get pooling operator in callable format
    
    """
    if isinstance(instance, str):
        supported_operator = get_default_pool_set()
        if instance in supported_operator:
            if instance == 'sum':
                return sum_
            elif instance == 'correlation1':
                return corr1_
            elif instance == 'correlation2':
                return corr2_
            elif instance == 'maximum':
                return max_
            else:
                raise "Pool names mismatch, given ´´%s´´, but only supported ´´%s´´" %(instance, supported_operator)
        else:
            raise "Pool operator ´´%s´´ not supported!"%instance
    
    else:
        assert callable(instance), "The given pooling operator is neither in default list nor callable"
        return instance
    
def get_activation_operator(instance):
    """Get activation operator in callable format
    
    """
    if isinstance(instance, str):
        supported_operator = get_default_activation_set()
        if instance in supported_operator:
            if instance == 'sigmoid':
                return sigmoid_
            elif instance == 'tanh':
                return tanh_
            elif instance == 'relu':
                return relu_
            elif instance == 'soft_linear':
                return soft_linear_
            elif instance == 'inverse_absolute':
                return inverse_absolute_
            elif instance == 'exp_linear':
                return exp_linear_
            else:
                raise "Activation names mismatch, given ´´%s´´, but only supported ´´%s´´" %(instance, supported_operator)
        else:
            raise "Activation operator ´´%s´´ not supported!"%instance
    
    else:
        assert callable(instance), "The given activation operator is neither in default list nor callable"
        return instance


def has_valid_operator(nodal_set, pool_set, activation_set):
    """Check if the names of operator are valid
    
    Args:
        nodal_set (tuple): names of nodal operator
        pool_set (tuple): names of pooling operator
        activation_set (tuple): names of activation operator
        
    Returns:
        
    
    """
    
    supported_nodal = get_default_nodal_set()
    supported_pool = get_default_pool_set()
    supported_activation = get_default_activation_set()
    
    for nodal in nodal_set:
        if isinstance(nodal, str):
            if not nodal in supported_nodal:
                raise 'Nodal operator ´´%s´´ not supported' % nodal
        else:
            assert callable(nodal), "The given nodal operator is neither in default list nor callable"
            

    for pool in pool_set:
        if isinstance(pool, str):
            if not pool in supported_pool:
                raise "Pooling operator ´´%s´´ not supported" % pool
        else:
            assert callable(pool), "The given pooling operator is neither in default list nor callable"

    for activation in activation_set:
        if isinstance(activation, str):
            if not activation in supported_activation:
                raise "Activation operator ´´%s´´ not supported" % pool
        else:
            assert callable(pool), "The given activation operator is neither in default list nor callable"
        
    return