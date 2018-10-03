.. _quickstart:

***********
Quickstart
***********


This library includes the implementation of the following models:

    * :ref:`pop-model`
    * :ref:`hemlgop-model`
    * :ref:`homlgop-model`
    * :ref:`hemlrn-model`
    * :ref:`homlrn-model`
    * :ref:`popfast-model`
    * :ref:`popmem-model` (POPmemO and POPmemH)

The library exposes a common interface for all models. Start by importing all the models::

    from GOP import models

To create an instance of a model, e.g., POP::

    model = models.POP()

Default parameters can be retrieved by calling :ref:`getdefaultparameters-function`::

    params = model.get_default_parameters()

For full description of parameters of each model, please refer to :ref:`algorithms`. This library adopts a particular data feeding mechanism that uses python generator. This design allows low memory usage even with large datasets, flexible and efficient user-defined preprocessing steps.

Generally, to feed the data to a model, the model accepts a **data_func** (a python function) and **data_arguments** (any type that is can be pickled)  with the assumption that a python generator and the number of mini-batches can be produced with syntax::

    data_gen, steps = data_func(data_arguments)

For example, if *data.pickle* contains *x_train* and *y_train* and *data.pickle* resides in directory *data_dir*, we can simply have **data_func** and **data_arguments** as follows::

    import numpy as np

    def data_func(data_arguments):
    """Define data function that loads pickled data from filename
       and yield mini-batch of batch_size

    """
        # 1st element from data_arguments is filename
        # 2nd element is batch size
        filename, batch_size = data_arguments
        
        # load pickled data from filename
        with open(filename, 'r') as fid:
            data = pickle.load(filename)

        x_train = data['x_train']
        y_train = data['y_train']

        # calculate number of mini-batches, suppose 1st dimension of x_train is #samples
        N = x_train_shape[0]
        steps = int(np.ceil( N / float(batch_size)))
        
        # give definition of generator

        def gen():
            while True:
                for step in range(steps):
                    start_idx = step*batch_size
                    stop_idx = min(N, (step+1)*batch_size)

                    yield x_train[start_idx:stop_idx], y_train[start_idx:stop_idx]


        return gen(), steps # note that gen() but not gen

    # Now define data_argument
    data_arguments = ['data_dir/data.pickle', 128]
    
With **data_func** and **data_argument**, we can fit the model by simply calling :ref:`fit-function` ::

    performance, progressive_history, finetune_history = model.fit(params, data_func, data_argument)

*performance* is a dictionary of best performances (loss and metrics), 
*progressive_history* contains all performances during progressive learning step and 
*finetune_history* contains all performances (at each epoch) during the fine-tuning step.

The trained model can be serialised and saved to disk with the given filename, e.g. 'pop_model.pickle' using :ref:`save-function`::

    model.save('pop_model.pickle') 

The pickled model can be loaded again later using :ref:`load-function`::

    model = models.POP()
    model.load('pop_model.pickle')

Using this trained model to evaluate test data e.g., *test_func* and *test_arguments* with new metrics, e.g. *mean_absolute_error*::
    
    metrics = ['mean_absolute_error',]
    performance = model.evaluate(test_func, test_arguments, metrics)
    # performance is a dictionary of a single key 'mean_absolute_error'

Or using this trained model to predict (:ref:`predict-function`) with unseen data e.g., *new_data_func*, *new_data_arugments*. Note that the generator produced by *new_data_func(new_data_arguments)* should only yield x but not (x,y)::

    prediction = model.predict(new_data_func, new_data_arguments)

Or finetune this trained model using :ref:`finetune-function` on potentially new training data and select best model settings through validation data and also report performances on test data ::

    history, performance = model.finetune(params, train_func, train_data, val_func, val_data, test_func, test_data)

While the above example is for POP, all other algorithms have the same interface, thus can be used in the same way. Different model, however, requires some specific parameters which should be consulted from :ref:`algorithms`. 

To configure computation environment (using CPU/GPU or using cluster), please refer :ref:`computation` 

For more discussion on data feeding mechanism, please refer :ref:`data`  

To deal with customization such as using custom loss, custom metrics or custom operators for nodal, pooling and activation, please refer :ref:`customization`


**Finally, it's worth noting that in case the script got interfered before completing the progressive learning step, PyGOP allows resuming to what has been learned as long as the 'tmp_dir' and 'model_name' in params have not been modified**


