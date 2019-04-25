.. _installation:

*************
Installation
*************

PyPi installation
=================
Tensorflow is required before installing PyGOP. 
To install tensorflow CPU version through *pip*::

    pip install tensorflow

Or the GPU version::
    
    pip install tensorflow-gpu

To install PyGOP with required dependencies::
    
    pip install pygop

At the moment, PyGOP only supports Linux with both python 2 and python 3 (tested on Python 2.7 and Python 3.4, 3.5, 3.6, 3.7 with tensorflow for cpu)

Installation from source
========================

To install latest version from github, clone the source from the project repository and install with setup.py::

    git clone https://github.com/viebboy/PyGOP
    cd PyGOP
    python setup.py install --user

