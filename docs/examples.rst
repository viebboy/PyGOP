.. _examples:

**********************
Illustrative Examples
**********************

In this section, we illustrate a complete usage of all algorithms through a hand-written digits recognition task and a face recognition task.

Hand-written Digits Recognition with Mnist dataset
=================================================

The complete example is available from `https://github.com/viebboy/PyGOP/tree/master/examples/train_mnist.py <https://github.com/viebboy/PyGOP/tree/master/examples/train_mnist.py>`_ . 'train_mnist.py' is the only source code we need, beside having PyGOP installed

Since we will use Mnist dataset available from Keras, we will simply create a data function by loading Mnist from keras and create a generator to generate mini-batches of data according to the batch size and decide whether to shuffle the data depending on the train or test set::

    def data_func(data_argument):
        """
        Data function of mnist for PyGOP models which should produce a generator and the number
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

        steps_per_epoch = int(np.ceil(N/ float(batch_size)))

        def gen():
            while True:
                indices = np.arange(N)
                # if train set, shuffle data in each epoch
                if split == 'train':
                    np.random.shuffle(indices)

                for step in range(steps_per_epoch):
                    start_idx = step * batch_size
                    stop_idx = min(N, (step + 1) * batch_size)
                    idx = indices[start_idx:stop_idx]
                    yield X[idx], Y[idx]

        # it's important to return generator object, which is gen() with the bracket
        return gen(), steps_per_epoch

It's worth noting that PyGOP will pickle *data_argument* passed to *data_func*, we should not pass the actual dataset through *data_argument* because it is computationally expensive. *data_argument* should ideally contain hyper-parameters of the generator only, such as *batch_size* and *split* above, or as you will see from the next example, a path to the actual data. In this way, we can perform the data reading step inside *data_func* to avoid many write and read operations of the actual data.

The main function then continues with parsing the two arguments: the name of the model and the type of computation (cpu/gpu)::

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

The type of model/algorithm is in *model_name* variable, and the type of computation is in *computation* variable. The main function continues with creating the corresponding model instance and setting the model's parameters::

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

To train the model instance, we simply call the *fit()* method from the model instance, using *train_func* as specified above::

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

In order to run the script using :ref:`hemlgop-model` algorithm, for example, with cpu, simply run the following command::

        python train_mnist.py -m hemlgop -c cpu

This will train the model with 8 parallel threads on cpu. The number of cpu threads or the gpu devices can be set within *train_mnist.py*



Face Recognition with CelebA dataset
====================================

The dataset is a small subset of `CelebA dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ including facial images of 20 identities, each having 100/30/30 train/validation/test images. We have extracted the deep features (using pretrained VGGface) to be used as input to all networks.


Preparation
-----------

To run this example, please fetch the examples directory from `https://github.com/viebboy/PyGOP/tree/master/examples <https://github.com/viebboy/PyGOP/tree/master/examples>`_

The examples directory includes the following files:

    * `prepare_miniCelebA.py <https://github.com/viebboy/PyGOP/blob/master/examples/prepare_miniCelebA.py>`_: this script loads raw images and generate deep features. However, we have extracted aand also provide the features, which can be downloaded via `this <https://drive.google.com/open?id=1njcxMypmE2n8VczvFWPMBG--rFpf0vsw>`_ 

    * `data_utility.py <https://github.com/viebboy/PyGOP/blob/master/examples/data_utility.py>`_: this script includes the data loading functionalities.

    * `train_miniCelebA.py <https://github.com/viebboy/PyGOP/blob/master/examples/train_miniCelebA.py>`_: the training script used for all algorithms.


To run this example, it suffices to just download `miniCelebA_deepfeatures.tar.gz <https://drive.google.com/open?id=1njcxMypmE2n8VczvFWPMBG--rFpf0vsw>`_ and extract it to the same folder as `train_miniCelebA.py <https://github.com/viebboy/PyGOP/blob/master/examples/train_miniCelebA.py>`_ and `data_utility.py <https://github.com/viebboy/PyGOP/blob/master/examples/data_utility.py>`_.

However, readers who want to do the data extraction process by their own can download the raw data `miniCelebA.tar.gz <https://drive.google.com/open?id=17Zax2B5NO0ZiyFGBpmd1QplIPu_oEdx0>`_ and extract the data in the same example folder. After that, running `prepare_miniCelebA.py <https://github.com/viebboy/PyGOP/blob/master/examples/prepare_miniCelebA.py>`_ will generate the deep features in data directory in the same directory. Since `prepare_miniCelebA.py <https://github.com/viebboy/PyGOP/blob/master/examples/prepare_miniCelebA.py>`_ requires a package called keras_vggface, which uses an older version of keras. It is advised to create a new environment when running this data preparation script to prevent breaking your current setup of keras. 

Usage
-----

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
