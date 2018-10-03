# PyGOP: A Python library for Generalized Operational Perceptron (GOP) based algorithms
[![Documentation Status](https://readthedocs.org/projects/pygop/badge/?version=latest)](http://pygop.readthedocs.io/en/latest/?badge=latest)

This package implements progressive learning algorithms using [Generalized Operational Perceptron](https://www.sciencedirect.com/science/article/pii/S0925231216312851). PyGOP supports both single machine and cluster environment using CPU or GPU. This implementation includes the following algorithms:

* Progressive Operational Perceptron ([POP](https://www.sciencedirect.com/science/article/pii/S0925231216312851))
* Heterogeneous Multilayer Generalized Operational Perceptron ([HeMLGOP](https://arxiv.org/abs/1804.05093)) and its variants
* Fast Progressive Operational Perceptron ([POPfast](https://arxiv.org/abs/1808.06377)) 
* Progressive Operational Perceptron with Memory ([POPmemO](https://arxiv.org/abs/1808.06377), [POPmemH](https://arxiv.org/abs/1808.06377))




What is Generalized Operational Perceptron?
===========================================


[Generalized Operational Perceptron](https://www.sciencedirect.com/science/article/pii/S0925231216312851) is an artificial neuron model that was proposed to replace the traditional McCulloch-Pitts neuron model. While standard perceptron model only performs a linear transformation followed by non-linear thresholding, GOP model encapsulates a diversity of both linear and non-linear operations (with traditional perceptron as a special case). Each GOP is characterized by learnable synaptic weights and an operator set comprising of 3 types of operations: nodal operation, pooling operation and activation operation. The 3 types of operations performed by a GOP loosely resemble the neuronal activities in a biological learning system of mammals in which each neuron conducts electrical signals over three distinct operations:

* Modification of input signal from the synapse connection in the Dendrites.
* Pooling operation of the modified input signals in the Soma.
* Sending pulses when the pooled potential exceeds a limit in the Axon hillock.

By defining a set of nodal operators, pooling operators and activation operators, each GOP can select the suitable operators based on the problem at hand. Thus learning a GOP-based network involves finding the suitable operators as well as updating the synaptic weights. The author of GOP proposed Progressive Operational Perceptron (POP) algorithm to progressively learn GOP-based networks. Later, [Heterogeneous Multilayer Generalized Operational Perceptron (HeMLGOP)](https://arxiv.org/pdf/1804.05093.pdf) algorithm and its variants (HoMLGOP, HeMLRN, HoMLRN) were proposed to learn heterogeneous architecture of GOPs with efficient operator set search procedure. In addition, fast version of POP, i.e., [POPfast](https://arxiv.org/pdf/1808.06377.pdf) was proposed together with memory extensions [POPmemO](https://arxiv.org/pdf/1808.06377.pdf), [POPmemH](https://arxiv.org/pdf/1808.06377.pdf) that augment POPfast by incorporating memory path.

Installation
============

PyPi installation
-----------------

Tensorflow is required before installing PyGOP.
To install tensorflow CPU version through *pip*::

    pip install tensorflow

Or the GPU version::

    pip install tensorflow-gpu

To install PyGOP with required dependencies::

    pip install pygop

At the moment, PyGOP only supports Linux with both python 2 and python 3 (tested on Python 2.7 and Python 3.5)

Installation from source
------------------------

To install latest version from github, clone the source from the project repository and install with setup.py::

    git clone https://github.com/viebboy/PyGOP
    cd PyGOP
    python setup.py install --user
 

Documentation
=============

Full documentation can be found [here](https://pygop.readthedocs.io)


Reference
=========

If you use one of the algorithms, please cite the corresponding articles:

* S. Kiranyaz, T. Ince, A. Iosifidis and M. Gabbouj, "Progressive Operational Perceptron", Neurocomputing, vol 224, pp. 142-154, 2017.
* D. T. Tran, S. Kiranyaz, M. Gabbouj and A. Iosifidis, "Heterogeneous Multilayer Generalized Operational Perceptron", arXiv preprint arXiv:1804.05093, 2018.
* D. T. Tran, S. Kiranyaz, M. Gabbouj and A. Iosifidis, "Progressive Operational Perceptron with Memory", arXiv preprint arXiv:1808.06377, 2018.


