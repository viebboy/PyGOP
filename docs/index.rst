.. PyGOP documentation master file, created by
   sphinx-quickstart on Mon Oct  1 02:28:03 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyGOP's documentation!
==================================

*PyGOP* provides a reference implementation of existing algorithms using Generalized Operational Perceptron (GOP) based on Keras and Tensorflow library. The implementation adopts a user-friendly interface while allowing a high level of customization including user-defined operators, custom loss function, custom metric functions that requires full batch evaluation such as precision, recall or f1. In addition, PyGOP supports different computation environments (CPU/GPU) for both single machine and cluster using SLURM job scheduler. What's more? Since training GOP-based algorithms might take days, PyGOP allows resuming to what has been learned in case the script got interfered in the middle during the progression!

What is Generalized Operational Perceptron?
---------------------------------------------

Generalized Operational Perceptron (`GOP <https://www.sciencedirect.com/science/article/pii/S0925231216312851>`_) is an artificial neuron model that was proposed to replace the traditional McCulloch-Pitts neuron model. While standard perceptron model only performs a linear transformation followed by non-linear thresholding, GOP model encapsulates a diversity of both linear and non-linear operations (with traditional perceptron as a special case). Each GOP is characterized by learnable synaptic weights and an operator set comprising of 3 types of operations: nodal operation, pooling operation and activation operation. The 3 types of operations performed by a GOP loosely resemble the neuronal activities in a biological learning system of mammals in which each neuron conducts electrical signals over three distinct operations: 

    * Modification of input signal from the synapse connection in the Dendrites.
    * Pooling operation of the modified input signals in the Soma.
    * Sending pulses when the pooled potential exceeds a limit in the Axon hillock.

By defining a set of nodal operators, pooling operators and activation operators, each GOP can select the suitable operators based on the problem at hand. Thus learning a GOP-based network involves finding the suitable operators as well as updating the synaptic weights. The author of GOP proposed Progressive Operational Perceptron (POP) algorithm to progressively learn GOP-based networks. Later, `Heterogeneous Multilayer Generalized Operational Perceptron (HeMLGOP) <https://arxiv.org/pdf/1804.05093.pdf>`_ algorithm and its variants (HoMLGOP, HeMLRN, HoMLRN) were proposed to learn heterogeneous architecture of GOPs with efficient operator set search procedure. In addition, fast version of POP (`POPfast <https://arxiv.org/pdf/1808.06377.pdf>`_) was proposed together with memory extensions (`POPmemO <https://arxiv.org/pdf/1808.06377.pdf>`_, `POPmemH <https://arxiv.org/pdf/1808.06377.pdf>`_) that augment POPfast by incorporating memory path.

Documentation Overview
----------------------

* :ref:`changelog` documents major changes between versions.

* :ref:`installation` gives instruction on how to install PyGOP through pip or source.

* :ref:`quickstart` gives a brief introduction on how to use PyGOP interface.

* :ref:`examples` gives instruction on how to run our complete example code on a subset of CelebA dataset.

* :ref:`common-interface` is dedicated to common parameters and interface shared by all algorithms.

* :ref:`data` gives instruction on the data feeding mechanism of PyGOP.

* :ref:`computation` discusses how to setup parameters related to computation devices and computation environment such as single machine or cluster.

* :ref:`algorithms` gives brief description of each algorithm and algorithm-specific parameters.

* :ref:`customization` details how to define custom loss, metrics or operators.
 

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   quickstart
   example
   common
   data
   computation
   algorithms
   customization



