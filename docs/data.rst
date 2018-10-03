.. _data:

**********************
Data Feeding Mechanism
**********************

As briefly mentioned in :ref:`quickstart`, this library adopts a particular data feeding mechanism that requires the user to give a function that returns a *data generator* and *the number of steps*, in other words, *the number of mini-batches* in an epoch.

The reference about python generator can be found `here <https://wiki.python.org/moin/Generators>`_. Generally, a python generator is defined in a very similar way as a function, but with the *yield* statement. An example of a generator that takes 2 numpy arrays *X*, *Y* and produces mini-batch of size *batch_size* is given below::

    import numpy as np

    def example_generator(X, Y, batch_size, no_mb):
    """An example generator that takes 2 numpy arrays X, Y and batch size and number of mini-batches

    """

        # assume 1st dimension of X and Y is the number of samples
        N = X.shape[0]

        # this while statement allows this generator to generate data infinitely
        while True:
            for step in range(no_mb):
                start_idx = step*batch_size
                stop_idx = min(N, (step+1)*batch_size) # dont allow index out of range
                
                x = X[start_idx:stop_idx]
                y = Y[start_idx:stop_idx]

                """
                Potentially some processing steps here
                """

                yield x, y # produce pair (x,y)

Note that after generating data for *N* steps, the for loop finishes and the while loop continues to run new iteration. The sequence of *N* mini-batches is exactly the same for each iteration of the while loop (or each epoch). This behavior, however, should be avoided when using stochastic gradient descend methods. There should be randomness at each iteration in the way data is generated. Below is a slight modification of the *example_generator* that introduces randomness::

    import numpy as np
    import random

    def example_generator(X, Y, batch_size, no_mb):
    """An example generator that takes 2 numpy arrays X, Y and batch size and number of mini-batches
   
    """
        # assume 1st dimension of X and Y is the number of samples
        N = X.shape[0]

        # this while statement allows this generator to generate data infinitely
        while True:

            # generate the list of indices and shuffle
            indices = range(N)
            random.shuffle(indices)

            for step in range(no_mb):
                start_idx = step*batch_size
                stop_idx = min(N, (step+1)*batch_size) # dont allow index out of range
                
                x = X[indices[start_idx:stop_idx]]
                y = Y[indices[start_idx:stop_idx]]

                """
                Potentially some processing steps here
                """

                yield x, y # produce pair (x,y)

Using this definition of *data generator*, the user needs also to define a function that returns *data generators* and *the number of mini-batches*. Let assume that the data is stored on disk in a pickled format. We can write a simple *data_func* as follows::

    import pickle
    import numpy as np
    
    def data_func(filename):
    """An example of data_func that returns example_generator and number of batch
    """

        with open(filename, 'r') as fid:
            data = pickle.load(fid)

        # assume that X, Y is stored as elements in dictionary data
        X, Y = data['X'], data['Y']
        
        N = X.shape[0] # number of samples
        batch_size = 128 # size of mini-batch
        no_mb = int(np.ceil(N/float(batch_size))) # calculate number of mini-batches

        # get an instance of example_generator
        gen = example_generator(X, Y, batch_size, no_mb)

        # return generator and number of mini-batches
        return gen, no_mb

The above example of *data_func* takes the path to the data file, performs data loading, calculates the number of mini-batches and returns an instance of *example_generator* and *number of mini-batches*. 

**Since data_func and data_argument will be serialized and written to disk during computation, it is recommended to pass small parameters through data_argument such as filename. Although it is possible to pass the actual data as data_argument, doing so would incur overhead computation**



