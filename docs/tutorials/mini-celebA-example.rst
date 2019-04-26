.. _mini-celebA-example:

*************************************
Face Recognition with CelebA dataset
*************************************

The dataset is a small subset of `CelebA dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ including facial images of 20 identities, each having 100/30/30 train/validation/test images. We have extracted the deep features (using pretrained VGGface) to be used as input to all networks.


Preparation
============

To run this example, please fetch the examples directory from `https://github.com/viebboy/PyGOP/tree/master/examples <https://github.com/viebboy/PyGOP/tree/master/examples>`_

The examples directory includes the following files:

    * `prepare_miniCelebA.py <https://github.com/viebboy/PyGOP/blob/master/examples/prepare_miniCelebA.py>`_: this script loads raw images and generate deep features. However, we have extracted aand also provide the features, which can be downloaded via `this <https://drive.google.com/open?id=1njcxMypmE2n8VczvFWPMBG--rFpf0vsw>`_ 

    * `data_utility.py <https://github.com/viebboy/PyGOP/blob/master/examples/data_utility.py>`_: this script includes the data loading functionalities.

    * `train_miniCelebA.py <https://github.com/viebboy/PyGOP/blob/master/examples/train_miniCelebA.py>`_: the training script used for all algorithms.


To run this example, it suffices to just download `miniCelebA_deepfeatures.tar.gz <https://drive.google.com/open?id=1njcxMypmE2n8VczvFWPMBG--rFpf0vsw>`_ and extract it to the same folder as `train_miniCelebA.py <https://github.com/viebboy/PyGOP/blob/master/examples/train_miniCelebA.py>`_ and `data_utility.py <https://github.com/viebboy/PyGOP/blob/master/examples/data_utility.py>`_.

However, readers who want to do the data extraction process by their own can download the raw data `miniCelebA.tar.gz <https://drive.google.com/open?id=17Zax2B5NO0ZiyFGBpmd1QplIPu_oEdx0>`_ and extract the data in the same example folder. After that, running `prepare_miniCelebA.py <https://github.com/viebboy/PyGOP/blob/master/examples/prepare_miniCelebA.py>`_ will generate the deep features in data directory in the same directory. Since `prepare_miniCelebA.py <https://github.com/viebboy/PyGOP/blob/master/examples/prepare_miniCelebA.py>`_ requires a package called keras_vggface, which uses an older version of keras. It is advised to create a new environment when running this data preparation script to prevent breaking your current setup of keras. 

Usage
======

After preparing the necessary files and data, the example folder should hold at least the following content

* examples/data_utility.py
* examples/train_miniCelebA.py
* examples/data/miniCelebA_x_train.npy
* examples/data/miniCelebA_y_train.npy
* examples/data/miniCelebA_x_val.npy
* examples/data/miniCelebA_y_val.npy
* examples/data/miniCelebA_x_test.npy
* examples/data/miniCelebA_y_test.npy

The signature of train_miniCelebA.py is as follows::

    python train_miniCelebA.py -m <model name> -c <computation device (cpu/gpu)> 

For example, to train HeMLGOP on CPU, we simply run::

    python train_miniCelebA.py -m hemlgop -c cpu

or training POP on GPU, we simply run::

    python train_miniCelebA.py -m pop -c gpu

For CPU, we have configured the script to run 8 parallel processes, and for GPU we have configured the script to run on 4 GPUs. Please change the configuration inside `train_miniCelebA.py <https://github.com/viebboy/PyGOP/blob/master/examples/train_miniCelebA.py>`_ to suit your setup.

After completing the training process, the performance and time taken will be written in result.txt in the same folder.


