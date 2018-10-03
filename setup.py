#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author: Dat Tran, Alexandros Iosifidis
Email: dat.tranthanh@tut.fi, viebboy@gmail.com, iosifidis.alekos@gmail.com
github: https://github.com/viebboy
"""

import setuptools
from GOP.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()
    
    
setuptools.setup(
    name="pygop",
    version=__version__,
    author="Dat Tran, Alexandros Iosifidis",
    author_email="viebboy@gmail.com, iosifidis.alekos@gmail.com",
    description="Python package that implements various algorithms using Generalized Operational Perceptron",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/viebboy/PyGOP",
    license='LICENSE.txt',
    packages=setuptools.find_packages(),
    classifiers=['Operating System :: POSIX',],
    install_requires = ['python_version>= "2.7"' or 'python_version >= "3.4"',
                      'numpy >= 1.13',
                      'dill >= 0.2.6',
                      'joblib >= 0.11',
                      'keras >= 2.2.1'],
    setup_requires = ['numpy >= 1.13',
                      'dill >= 0.2.6',
                      'joblib >= 0.11',
                      'keras >= 2.2.1']
)
