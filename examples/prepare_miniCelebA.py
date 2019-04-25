#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from scipy import misc
from keras import Model
from keras.utils import to_categorical
from keras.layers import GlobalAveragePooling2D
import os
from glob import glob


def get_deep_feature(x):

    model = VGGFace(include_top=False, input_shape=(224, 224, 3),
                    pooling='avg')  # pooling: None, avg or max
    output = model.get_layer('conv5_3').output
    output = GlobalAveragePooling2D()(output)
    feature_model = Model(inputs=model.input, outputs=output)

    x = utils.preprocess_input(x, version=1)  # or version=2
    x = feature_model.predict(x)

    return x


def prepare_data(src, dst):
    """
    Function that extract VGGface features from raw images

    Args:
        - src (string): path to source folder, should include train, val and test as subfolders
        - dst (string): path to save the data

    Returns:

    """

    data_prefix = 'miniCelebA_'
    for split in ['train', 'val', 'test']:
        print('processing %s split' % split)
        if (not os.path.exists(os.path.join(dst, 'x_' + split + '.npy')) or not
                os.path.exists(os.path.join(dst, 'y_' + split + '.npy'))):
            labels = glob(os.path.join(src, split, '*'))
            no_sample = 0
            for lb in labels:
                no_sample += len(os.listdir(lb))

            x = np.zeros((no_sample, 224, 224, 3))
            y = np.zeros((no_sample, 20))
            count = 0
            for lb in labels:
                files = glob(os.path.join(lb, '*.png'))
                for f in files:
                    print('processing file: %s, with label %s' % (f, lb.split('/')[-1]))
                    y[count] = to_categorical(int(lb.split('/')[-1]), 20)
                    img = misc.imresize(misc.imread(f), (224, 224), 'bicubic')
                    if img.ndim == 2:
                        img = np.expand_dims(img, -1)
                        img = np.concatenate((img, img, img), axis=-1)
                    x[count] = img

                    count += 1

            assert count == no_sample, "number of sample (%d) is different than number of read image (%d)" % (
                no_sample, count)

            x = get_deep_feature(x)
            np.save(os.path.join(dst, data_prefix + 'x_' + split + '.npy'), x)
            np.save(os.path.join(dst, data_prefix + 'y_' + split + '.npy'), y)


src = 'miniCelebA'
dst = 'data'

prepare_data(src, dst)
