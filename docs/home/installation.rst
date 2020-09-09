.. _installation:

*************
Installation
*************

PyPi installation
=================
Tensorflow version 1 is required before installing PyGOP. We suggest tensorflow 1.14.0 for efficiency 
To install tensorflow CPU version through *pip*::

    pip install tensorflow==1.14.0

Or the GPU version::
    
    pip install tensorflow-gpu==1.14.0

To install PyGOP with required dependencies::
    
    pip install pygop

At the moment, PyGOP only supports Linux with python 2 and python 3 (tested on Python 2.7 and Python 3.5, 3.6, 3.7 with tensorflow for cpu)

Installation from source
========================

To install latest version from github, clone the source from the project repository and install with setup.py::

    git clone https://github.com/viebboy/PyGOP
    cd PyGOP
    python setup.py install --user

