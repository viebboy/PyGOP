.. _customization:

**************
Customization
**************

This section describes how to customize PyGOP to your need!

.. _custom-loss:

Custom loss
===========

The loss function is specified through **params['loss']** (see :ref:`common-parameter`), which is 'mse' (mean square error) by default. All the loss functions defined by Keras is supported in PyGOP. These includes:

* *'mean_squared_error'*
* *'mean_absolute_error'*
* *'mean_absolute_percentage_error'*
* *'mean_squared_logarithmic_error'*
* *'squared_hinge'*
* *'hinge'*
* *'categorical_hinge'*
* *'logcosh'*
* *'categorical_crossentropy'*
* *'sparse_categorical_crossentropy'*
* *'binary_crossentropy'*
* *'kullback_leibler_divergence'*
* *'poisson'*
* *'cosine_proximity'*

The above strings can be set to **params['loss']** to indicate the loss function. User can also define the definition of the loss function and assign the callable to **params['loss']**. A custom loss function should follow this template

.. code-block:: python

    def cutom_loss(y_true, y_pred):
        # calculation of the loss using tensorflow or keras backend operations
        
        return loss # loss should be a scalar

Note that the computation in the loss function must be expressed by tensorflow or keras operations. Below gives an example of a custom mean squared error that only calculate the error if two corresponding elements have the same sign

.. code-block:: python

    """An example of custom loss function for PyGOP models
    """
    import tensorflow as tf

    def custom_mse(y_true, y_pred):
    # assume 1st dimension is the number of samples
        mask = y_true * y_pred > 0
        mse = tf.reduce_sum(tf.flatten(mask * (y_true - y_pred)**2))

        return mse


    # set the custom loss to the dictionary of model parameters
    params['loss'] = custom_mse

.. _custom-metrics:

Custom metrics
==============

This refers to **params['metrics']**, which are standard metrics that can be accumulated over mini-batches. Similar to loss function, we can also define a list of metrics that has both built-in metrics and custom metrics. Note that all the loss strings listed above can also be used as metrics. In addition, Keras has the following built-in metrics:

* *'binary_accuracy'*
* *'categorical_accuracy'*
* *'sparse_categorical_accuracy'*
* *'top_k_categorical_accuracy'*
* *'sparse_top_k_categorical_accuracy'*

The custom metric can be defined in exactly the same manner as custom loss. Suppose we need to monitor 'accuracy', mean_squared_error and 'custom_mse' defined above, we can set metrics as follows:

.. code-block:: python

    """An example that defines custom metric for PyGOP models
    """
    import tensorflow as tf

    def custom_mse(y_true, y_pred):
    # assume 1st dimension is the number of samples
        mask = y_true * y_pred > 0
        mse = tf.reduce_sum(tf.flatten(mask * (y_true - y_pred)**2))

        return mse

    # set the metrics to the dictionary of model parameters
    params['metrics'] = ['accuracy', 'mean_squared_error', custom_mse]

.. _special-metrics:

Special Metrics
===============

This refers to **params['special_metrics']**, which categorizes those metrics that require y_true and y_pred of the full batch to evaluate. Special metrics are given as a list of user-defined functions, which use should take **numpy arrays** as input to compute the metrics. Examples of special metrics are precision, recall or f1. Below gives an example that defines average precision, average recall and average f1 as special metrics:

.. code-block:: python

    """An example that defines average precision, recall and f1 using sklearn metrics 
       and use this metrics as special metrics in PyGOP
    """
    from sklearn import metrics

    def custom_precision(y_true, y_pred):
    # assume 1st dimension is the number of samples
        y_true_lb = np.argmax(y_true, axis=-1)
        y_pred_lb = np.argmax(y_pred, axis=-1)

        return metrics.f1_score(y_true_lb, y_pred_lb, average='macro')

    def custom_recall(y_true, y_pred):
    # assume 1st dimension is the number of samples
        y_true_lb = np.argmax(y_true, axis=-1)
        y_pred_lb = np.argmax(y_pred, axis=-1)

        return metrics.f1_score(y_true_lb, y_pred_lb, average='macro')

    def custom_f1(y_true, y_pred):
    # assume 1st dimension is the number of samples
        y_true_lb = np.argmax(y_true, axis=-1)
        y_pred_lb = np.argmax(y_pred, axis=-1)

        return metrics.f1_score(y_true_lb, y_pred_lb, average='macro')

    # set special metrics to the dictionary of model paramters
    params['special_metrics'] = [custom_precision, custom_recall, custom_f1]


**If params['convergence_measure'] is one of the custom metrics or special metrics, it should be specified as the name of the function, e.g. params['convergence_measure'] = 'custom_f1'**

.. _custom-operators:

Custom Operators
================

While PyGOP specifies built-in library of operators as defined in Table 1 in `here <https://arxiv.org/pdf/1804.05093.pdf>`_, custom operators can be defined by users and given to train a model. All custom operators must be implemented using tensorflow or keras operators. Below gives the templates on how to define custom nodal, pooling or activation operators that can be used by PyGOP

.. code-block:: python

    """Template of custom operators
    """

    def custom_nodal(x, w):
    """Description of custom nodal operator format
    
    All nodal operators must take as input two tensors x and w, which are the input and weights
    x and w are assumed to have compatible shape so that the element-wise multiplication (x*w) is valid
    
    All nodal operations should be element-wise operation, meaning that each individual input signal x[i] is
    modified by the corresponding weight element w[i]

    Here we give as an example the 'multiplication' operator
    
    """

    return x*w


    def custom_pool(z):
    """Description of the pooling operator format
    
    All pooling operators must take only one input z, which is the output of the nodal operation
    z has specific shape of [N, D, d]
    - N is the number of samples
    - D is the number of neurons in the previous layer (number of input signals)
    - d is the number of neurons in the current layer

    The pooling operation performs pooling over D input signals, thus the pooling is performed on axis=1
    The output y of the pooling operator should has shape [N, d]

    Here we give as an example of the 'sum' operator

    """

    y = tf.reduce_sum(z, axis=1)
    return y

    def custom_activation(y):
    """Description of the activation operator format

    All activation operators should be element-wise operation. 
    Here we give as an example the 'sigmoid' operator
    """

    return tf.sigmoid(y)

After defining the custom operators, these functions can be included in the list of **params['nodal_set']**, **params['pool_set']**, **params['activation_set']** together with/without other built-in operators. See :ref:`common-parameter` for the default built-in values 
