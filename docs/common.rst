.. _common-interface:

*****************
Common Interface
*****************


.. _common-parameter:

Common parameters
=================

Parameters are given to a model through a dictionary **params**. While different model has different model-specific parameters, there are some common parameters that possessed by all models. Parameters are given as (key,value) pairs in the dictionary **params**. Description of common parameters are below

* **tmp_dir**: String of temporary directory used during computation, compulsory parameter
* **model_name**: String of the current model instance. This allows several model instances sharing the same **tmp_dir**, compulsory parameter
* **input_dim**: Int that specifies the input dimension of the data, compulsory parameter
* **output_dim**: Int that specifies the output dimension of the data, compulsory parameter
* **nodal_set**: List of (string/callable) that specify nodal operators, default ['multiplication','exponential','harmonic','quadratic','gaussian','dog']
* **pool_set**: List of (string/callable) that specify pool operators, default ['sum','correlation1','correlation2','maximum']
* **activation_set**: List of (string/callable) that specify activation operators, default ['sigmoid','relu','tanh','soft_linear', 'inverse_absolute', 'exp_linear']
* **metrics**: List of metrics, each metric is either a string indicating the metric function supported by Keras or a callable defined by using Keras/Tensorflow operations, default ['mse',]. See :ref:`custom-metrics`
* **special_metrics**: List of special metrics, each metric is a callable that takes 2 numpy arrays (y_true, y_pred). Special metrics are those that requires full batch evaluation such as Precision, Recall or F1, default None. See :ref:`special-metrics`
* **loss**: Loss function, either a string indicating loss function supported by Keras or a callable defined by using Keras/Tensorflow operations, default 'mse'. See :ref:`custom-loss`
* **convergence_measure**: String indicates which metric value to monitor the stopping criterion and to gauge the performance when choosing operator sets and weights. **convergence_measure** should also belong to either **metrics** or **special_metrics**, default 'mse'.
* **direction**: String indicates how to compare two measures. 'higher' means the higher the better, 'lower' means the lower the better. This should be set in accordance with **convergence_measure**, e.g., if **convergence_measure** is *mean_square_error* then **direction** should be 'lower' to indicate that lower values of mean_square_error are better, default 'lower'.
* **direct_computation**: True if perform computation on full batch when possible, otherwise computation is done in a mini-batch manner. For large dataset, it is recommended to set False, default False
* **search_computation**: Tuple with 1st element indicating computation devices during the search procedure. 1st element should be either 'cpu' or 'gpu'. If using 'gpu' then 2nd element should be a list(int) of gpu devices, default ('cpu',). See :ref:`computation` for detail description.
* **finetune_computation**: Tuple with 1st element indicating computation devices during the finetune procedure. 1st element should be either 'cpu' or 'gpu'. If using 'gpu' then 2nd element should be a list(int) of gpu devices, default ('cpu',). See :ref:`computation` for detail description.
* **use_bias**: Bool indicates whether to use bias in the weights, default True
* **output_activation**: String indicates optional activation function (supported by Keras) for output layer, default None.
* **input_dropout**: Float indicates dropout percentage applied to input layer during Back Propagation, default None.
* **dropout**: Float indicates dropout percentage applied to hidden layers during Back Propagation in progressive learning step, default None.
* **dropout_finetune**: Float indicates dropout percentage applied to hidden layers during Back Propagation in finetuning step, default None.
* **weight_regularizer**: Float weight decay coefficient used during Back Propagation in progressive learning step, default None
* **weight_regularizer_finetune**: Float weight decay coefficient used during Back Propagation in finetuning step, default None
* **weight_constraint**: Float max-norm constraint value used during Back Propagation in progressive learning step, default None
* **weight_constraint_finetune**: Float max-norm constraint value used during Back Propagation in finetuning step, default None
* **optimizer**: String indicates the name of the optimizers implemented by Keras, default 'adam'
* **optimizer_parameters**: A dictionary to supply non-default parameters for the optimizer, default to None which means using default parameters of Keras optimizer
* **lr_train**: List of learning rates values in a schedule, default [0.01, 0.001, 0.0001]
* **epoch_train**: List of number of epochs for each learning rate value in **lr_train**, default [2,2,2]
* **lr_finetune**: List of learning rates values in a schedule, default [0.0005,]
* **epoch_finetune**: List of number of epochs for each learning rate value in **lr_train**, default [2,]
* **cluster**: Bool indicates if using SLURM cluster to compute. See :ref:`cluster-computation` for details using computation on a cluster.
* **class_weight**: Dict containing the weights given to each class in the loss function, default None. This allows weighing loss values from different classes

Refer :ref:`customization` when custom loss, custom metrics or operators


Below describes common interface implemented by all models.


.. _fit-function:

fit
===
.. code-block:: python

    fit(params, train_func, train_data, val_func=None, val_data=None, test_func=None, test_data=None, verbose=False)

Fits the model with the given parameters and data, this function perform :ref:`progressivelearn-function` to learn the network architecture and :ref:`finetune-function` to finetune the whole architecture. *Note that when validation data is available, the model weights selection and convergence criterion is measured on validation data, otherwise on train data* 


Arguments:

* **params**: Dictionary of model parameters. Consult above section :ref:`common-parameter` and :ref:`algorithms` for details of each model
* **train_func**: Callable that produces train data generator and the number of mini-batches. See :ref:`data`
* **train_data**: Input to **train_func** See :ref:`data`
* **val_func**: Callable that produces validation data generator and the number of mibi-batches, default None. See :ref:`data`
* **val_data**: Input to **val_func**, default None. See :ref:`data`
* **test_func**: Callable that produces test data generator and the number of mibi-batches, default None. See :ref:`data`
* **test_data**: Input to **test_func**, default None. See :ref:`data`
* **verbose**: Bool to indicate verbosity or not, default False.

Returns:

* **performance**: Dictionary that holds best performances with keys are loss, metrics and special metrics defined in **params**
* **p_history**: List of full history during progressive learning, with **p_history** [layer_idx][block_idx] is a dictionary similar to **performance**
* **f_history**: Dictionary of full history during finetuning

.. _progressivelearn-function:

progressive_learn
==================
.. code-block:: python

    progressive_learn(params, train_func, train_data, val_func=None, val_data=None, test_func=None, test_data=None, verbose=False)

Progressively learn the network architecture according to specific algorithm specified by each model. *Note that when validation data is available, the model weights selection and convergence criterion is measured on validation data, otherwise on train data*

Arguments:

* **params**: Dictionary of model parameters. Consult above section :ref:`common-parameter` and :ref:`algorithms` for details of each model
* **train_func**: Callable that produces train data generator and the number of mini-batches. See :ref:`data`
* **train_data**: Input to **train_func** See :ref:`data`
* **val_func**: Callable that produces validation data generator and the number of mibi-batches, default None. See :ref:`data`
* **val_data**: Input to **val_func**, default None. See :ref:`data`
* **test_func**: Callable that produces test data generator and the number of mibi-batches, default None. See :ref:`data`
* **test_data**: Input to **test_func**, default None. See :ref:`data`
* **verbose**: Bool to indicate verbosity or not, default False.

Returns:

* **history**: List of full history during progressive learning, with **history** [layer_idx][block_idx] is a dictionary with keys are loss, metrics and special metrics defined in **params**

.. _finetune-function:

finetune
========
.. code-block:: python

    finetune(params, train_func, train_data, val_func=None, val_data=None, test_func=None, test_data=None, verbose=False)

Finetune the whole network architecture, this required a trained model data exists either by calling *load()* or *fit()* or *progressive_learn()*. *Note that when validation data is available, the model weights selection and convergence criterion is measured on validation data, otherwise on train data*

Arguments:

* **params**: Dictionary of model parameters. Consult above section :ref:`common-parameter` and :ref:`algorithms` for details of each model
* **train_func**: Callable that produces train data generator and the number of mini-batches. See :ref:`data`
* **train_data**: Input to **train_func** See :ref:`data`
* **val_func**: Callable that produces validation data generator and the number of mibi-batches, default None. See :ref:`data`
* **val_data**: Input to **val_func**, default None. See :ref:`data`
* **test_func**: Callable that produces test data generator and the number of mibi-batches, default None. See :ref:`data`
* **test_data**: Input to **test_func**, default None. See :ref:`data`
* **verbose**: Bool to indicate verbosity or not, default False.

Returns:

* **history**: List of full history during progressive learning, with **history** [layer_idx][block_idx] is a dictionary with keys are loss, metrics and special metrics defined in **params**
* **performance**: Dictionary of best performances with keys are loss, metrics and special metrics defined in **params**

.. _evaluate-function:

evaluate
========
.. code-block:: python

    evaluate(data_func, data_argument, metrics, special_metrics=None, computation=('cpu',))

Evaluate the model with given data and metrics

Arguments:

* **data_func**: Callable that produces data generator and the number of mini-batches
* **data_argument**: Input to **data_func**
* **metrics**: List of metrics, with each metric can be computed through aggregation of evaluation on mini-batches, e.g., accuracy, mse
* **special_metrics**: List of special metrics, which can only be computed over full batch, e.g., f1, precision or recall
* **computation**: Tuple with 1st element is a string to indicate 'cpu' or 'gpu'. In case of 'gpu', 2nd element is a list of int which specifies gpu devices

Returns:

* **performance**: Dictionary of performances with keys are the metric names in **metrics** and **special_metrics**

.. _predict-function:

predict
=======
.. code-block:: python

    predict(data_func, data_argument, computation=('cpu',))

Using current model instance to generate prediction

Arguments:

* **data_func**: Callable that produces data generator and the number of mini-batches
* **data_argument**: Input to **data_func**
* **computation**: Tuple with 1st element is a string to indicate 'cpu' or 'gpu'. In case of 'gpu', 2nd element is a list of int which specifies gpu devices

Returns:

* **pred**: Numpy array of prediction

.. _save-function:

save
====
.. code-block:: python

    save(filename)

Save the current model instance to disk

Arguments:

* **filename**: String that specifies the name of pickled file

Returns:

.. _load-function:

load
====
.. code-block:: python

    load(filename)

Load a pretrained model instance from disk

Arguments:

* **filename**: String that specifies the name of pickled file

Returns:

.. _getdefaultparameters-function:

get_default_paramters
=====================
.. code-block:: python

    get_default_parameters()

Get the default parameters of the model

Arguments:

Returns:

* **params**: Dictionary of default parameters


