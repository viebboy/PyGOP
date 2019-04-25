#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This is an example showing how to train models in PyGOP using mnist dataset from Keras


Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import os
import time
import getopt
import sys
from GOP import models
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import random


def data_func(data_argument):
    """ Data function of mnist for PyGOP models which should produce a generator and the number
    of steps per epoch

    Args:
        data_argument: a tuple of batch_size and split ('train' or 'test')

    Return:
        generator, steps_per_epoch

    """

    batch_size, split = data_argument

    # load dataset from keras datasets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if split == 'train':
        X = x_train
        Y = y_train
    else:
        X = x_test
        Y = y_test

    # reshape image to vector
    X = np.reshape(X, (-1, 28 * 28))
    # convert to one-hot vector of classes
    Y = to_categorical(Y, 10)
    N = X.shape[0]

    steps_per_epoch = int(np.ceil(N / float(batch_size)))

    def gen():
        while True:
            indices = list(range(N))
            # if train set, shuffle data in each epoch
            if split == 'train':
                random.shuffle(indices)

            for step in range(steps_per_epoch):
                start_idx = step * batch_size
                stop_idx = min(N, (step + 1) * batch_size)
                idx = indices[start_idx:stop_idx]
                yield X[idx], Y[idx]

    # it's important to return generator object, which is gen() with the bracket
    return gen(), steps_per_epoch


def main(argv):

    try:
        opts, args = getopt.getopt(argv, "m:c:")
    except getopt.GetoptError:
        print('train_mnist.py -m <model> -c <computation option cpu/gpu>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-m':
            model_name = arg
        if opt == '-c':
            computation = arg

    # input 728 raw pixel values
    # output 10 class probability
    input_dim = 28 * 28
    output_dim = 10

    if computation == 'cpu':
        search_computation = ('cpu', 8)
        finetune_computation = ('cpu', )
    else:
        search_computation = ('gpu', [0, 1, 2, 3])
        finetune_computation = ('gpu', [0, 1, 2, 3])

    if model_name == 'hemlgop':
        Model = models.HeMLGOP
    elif model_name == 'homlgop':
        Model = models.HoMLGOP
    elif model_name == 'hemlrn':
        Model = models.HeMLRN
    elif model_name == 'homlrn':
        Model = models.HoMLRN
    elif model_name == 'pop':
        Model = models.POP
    elif model_name == 'popfast':
        Model = models.POPfast
    elif model_name == 'popmemo':
        Model = models.POPmemO
    elif model_name == 'popmemh':
        Model = models.POPmemH
    else:
        raise Exception('Unsupported model %s' % model_name)

    # create model
    model = Model()
    model_name += '_mnist'

    # get default parameters and assign some specific values
    params = model.get_default_parameters()

    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    params['tmp_dir'] = tmp_dir
    params['model_name'] = model_name
    params['input_dim'] = input_dim
    params['output_dim'] = output_dim
    params['metrics'] = ['acc', ]
    params['loss'] = 'categorical_crossentropy'
    params['output_activation'] = 'softmax'
    params['convergence_measure'] = 'acc'
    params['direction'] = 'higher'
    params['search_computation'] = search_computation
    params['finetune_computation'] = finetune_computation
    params['output_activation'] = 'softmax'
    params['input_dropout'] = 0.2
    params['weight_constraint'] = 3.0
    params['weight_constraint_finetune'] = 3.0
    params['optimizer'] = 'adam'
    params['lr_train'] = (1e-3, 1e-4, 1e-5)
    params['epoch_train'] = (60, 60, 60)
    params['lr_finetune'] = (1e-3, 1e-4, 1e-5)
    params['epoch_finetune'] = (60, 60, 60)
    params['direct_computation'] = False
    params['max_block'] = 5
    params['block_size'] = 40
    params['max_layer'] = 4
    params['max_topology'] = [200, 200, 200, 200]

    batch_size = 64
    start_time = time.time()

    performance, _, _ = model.fit(params,
                                  train_func=data_func,
                                  train_data=[batch_size, 'train'],
                                  val_func=None,
                                  val_data=None,
                                  test_func=data_func,
                                  test_data=[batch_size, 'test'],
                                  verbose=True)

    stop_time = time.time()

    with open('mnist_result.txt', 'a') as fid:
        fid.write('Finish training %s in %.2f seconds with following performance: \n' % (model_name,
                                                                                         stop_time - start_time))

        for key in performance.keys():
            if performance[key] is not None:
                fid.write('%s: %.4f \n' % (key, performance[key]))


if __name__ == "__main__":
    main(sys.argv[1:])
