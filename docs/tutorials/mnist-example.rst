.. _mnist-example:

****************************************************
Hand-written Digits Recognition with Mnist dataset
****************************************************

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
