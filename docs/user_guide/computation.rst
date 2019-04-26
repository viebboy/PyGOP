.. _computation:

************************
Computation Environments
************************

*computation* is specified through a tuple. The first element is a string, either 'cpu' or 'gpu', which indicates computation devices. 

* If 'gpu' is specified, the 2nd element of the tuple must be a list of int indicating the GPU device numbers. 
* If 'cpu' is specified, the 2nd element should be an integer indicating the number of parallel processes to run during operator set search (ideally it should be the number of cores on the local machine). This number, however, has no effect when doing finetuning since a single process will use the available cores. 

*computation* can be specified in the following cases:

* In **params** (dictionary) (see :ref:`common-parameter`) argument that is given to :ref:`fit-function`, :ref:`finetune-function`:

    * *search_computation*: this key specifies the computation device during the operator set search procedure.
    * *finetune_computation*: this key specifies the computation device during the finetuning step.

* *computation* is also an argument to :ref:`evaluate-function`, :ref:`predict-function`


*Note 1: depending on the computation setup available, 'search_computation' must be carefully set. If only tensorflow-gpu is installed on the system but 'search_computation' is set to 'cpu' with K processes, the library will attempt to launch K different processes trying to use all GPUs without explicit device allocation, which can lead to out of memory situation*


.. _local-cpu:

Local CPU
=========

This is the default computation option in all models. That is::

    params['search_computation'] = ('cpu', 8)
    params['finetune_computation'] = ('cpu',)
    evaluate(..., computation=('cpu',))
    predict(..., computation=('cpu',))

.. _local-gpu:

Local GPU
=========

Sometimes it is desirable to mix CPU and GPU computation when both versions of tensorflow are available. For example, in HeMLGOP or its variants (HoMLGOP, HeMLRN, HoMLRN), the search procedure solves a least square problem with computation implemented on CPU. Thus it might be desirable to use CPU during the search procedure to avoid data transfer between CPU and GPU and use GPU during the finetuning step. This can be done by setting the *computation* as follows::

    params['search_computation'] = ('cpu', K) # with K is the number of cores on the system
    params['finetune_computation'] = ('gpu', [0,1]) # finetuning with 2 GPUs

However, with algorithms relying only on Back Propagation during the search procedure such as POP, POPfast or POPmemO and POPmemH, it is desirable to use GPUs to perform the operator set search procedure::

    # using 4 GPUs to perform operator set searching
    # the implementation launches 4 processes, each of which sees only 1 GPU device
    params['search_computation'] = ('gpu', [0,1,2,3])

.. _cluster-computation:

Cluster Computation
===================

The library also supports running the operator set searching procedure on a SLURM cluster. Since the search procedure involves evaluating each operator set independently, which is highly parallelizable. Thus, on a cluster running SLURM, the user can instruct the library to run the search procedure on many machines by submitting new batch jobs. Note that this assumes the all nodes share the same disk system since they will try to access *tmp_dir/model_name* as specified in **params** (see :ref:`common-parameter`)


To allow the search procedure to submit new batch jobs, set ::

    params['cluster'] = True

In addition, a dictionary called **batchjob_parameters** that contains the configuration of the batch job file must be given to **params** as::

    params['batchjob_parameters'] = batchjob_parameters

**batchjob_parameters** must have the following (key, value):

* **name**: String that specifies the name of the job. This corresponds to option *#SBATCH -j* 
* **mem**: Int that specifies the amount of memory (or RAM) requested for each node in GB. This corresponds to option *#SBATCH --mem*
* **core**: Int that specifies the number of cores requested for each node. This corresponds to option *#SBATCH -c*
* **partition**: String that specifies the name of the partition to request the nodes. This corresponds to option *#SBATCH -p*
* **time**: String that specifies the maximum allowed time for each node, e.g., '2-00:00:00' indicates 2 days. This corresponds to option *#SBATCH -t*
* **no_machine**: Int that specifies the number of parallel nodes requested. e.g., if no_machine=4 and there are 72 operator sets, each node will process 18 different operator sets. This corresponds to option *#SBATCH -array*
* **python_cmd**: String that specifies the command how to run python on bash. In many cases, it is simply just 'python' if python is in the $PATH. In some systems, this involves calling 'srun python'


In addition, two optional keys can be set to allow specific configurations:

* **constraint**: String that specifies the constraint on the node, e.g. 'hsw' might indicate only request for Haswell architectures or 'gpu' only request for GPU nodes. This corresponds to option *#SBATCH --constraint*
* **configuration**: String that specifies all the necessary setup befores launching a python script on a node, e.g., this can be the setup of $PATH or module load, etc.

*Note that similar to the local case, both CPU and GPU can be used during the search procedure using the cluster. However,* **batchjob_parameters** *must be carefully set in accordance with all computation parameters setup*

* *If* **params['search_computation']** *indicate CPU*, **batchjob_parameters** *must be set so that the requested nodes and its configuration allow running tensorflow cpu version.* 
* *If* **params['search_computation']** *indicate GPU*, **batchjob_parameters** *must be set so that the requested nodes allow the access to the specified GPU device list and tensorflow-gpu can be invoked*

*In addition, the main script, which creates a model instance and operates on the model instance, is usually run on a node on the cluster, so ensure that* **params['finetune_computation']** *and other computation arguments used in evaluate(), predict() are set in accordance with the node configuration itself*


