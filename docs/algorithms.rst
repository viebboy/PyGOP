.. _algorithms:

**********
Algorithms
**********

This section briefly describes each algorithm implemented in the library as well as detail description of algorithm-specific parameters.

.. _pop-model:

`Progressive Operational Perceptron (POP) <https://www.sciencedirect.com/science/article/pii/S0925231216312851>`_
=================================================================================================================

Description
-----------

POP defines a maximum network template, which specifies the maximum number of layers and the size of each layer. The algorithm then learns one hidden layer at a time. All GOP neurons in the same layer share the same operator set and all layers are GOP layers, including the output layer. When learning a new hidden layer, all previous layers that have been learned are fixed. POP uses a two-pass Greedy Iterative Search (GIS) algorithm:

* Randomly select operator set (nodal, pool, activation) for the new hidden layer, let say *h*. For every operator set *op* in the library, assign *op* to the output layer and train the network (new hidden layer, output layer) for E epochs. Select the best performing operator set *o-1* according to **params['convergence_measure']** for the output layer.

* Fix *o-1* for the output layer, for every operator set *op* in the library, assign *op* to the new hidden layer and train the network for E epochs. Select the best operator set *h-1* for the hidden layer.

* Repeat step 1, however with *h-1* as the operator set for the hidden layer. This produces *o-2* for the output layer. 

* Repeat step 2, however with *o-2* as the operator set for the output layer. This produces *h-2*. 

GIS outputs *h-2* and *o-2* as the operator set for the hidden and output layer with current synaptic weights. The algorithm continues adding new hidden layer when the performance stops improving, given a threshold. After learning the architecture of the network (number of layer and operator set assignments), POP finetunes the synaptic weights of the whole architecture using Back Propagation.


Parameters
----------

The following parameters are specific to POP that should be set in **params** dictionary:

* **max_topology**: List of ints that specifies the maximum network topology, default [40,40,40,40], which means 4 hidden layers with 40 GOPs each
* **layer_threshold**: Float that specifies the threshold on the relative improvement, default 1e-4. See equation (8) in `here <https://arxiv.org/pdf/1804.05093.pdf>`_ for more details.

.. _hemlgop-model:

`Heterogeneous Multilayer Operational Perceptron (HeMLGOP) <https://arxiv.org/pdf/1804.05093.pdf>`_
====================================================================================================

Description
-----------

HeMLGOP learns a heterogeneous layers of GOPs in a block-wise manner. At each step, the algorithm adds a new block to the current hidden layer by searching for the suitable operator set and its synaptic weights. When the performance saturates in the current hidden layer, HeMLGOP constructs a new hidden layer composed of one block. The progression in the new hidden layer continues until reaching saturation. HeMLGOP then evaluates if adding this new hidden layer improves the performance. The algorithm terminates when adding new hidden layer stops improving the performance, given a threshold. 

Different than POP, HeMLGOP assumes linear output layer. To evaluate an operator set assignment to the new block, HeMLGOP assigns random weights to the new block and optimizes the weights of the output layer by solving a least-square problem. After selecting the best operator set for the new block, HeMLGOP performs the weight update of the new block and output layer through Back Propagation. Once a block is learned, it is fixed (operator set assignment and weights). Similar to POP, once the network architecture is learned in the progressive learning step, HeMLGOP finetunes all the weights in the network. 

Parameters
----------

The following parameters are specific to HeMLGOP that should be set in **params** dictionary:

* **block_size**: Int that specifies the number of neurons in a new block, default 20
* **max_block**: Int that specifies the maximum number of blocks in a hidden layer, default 5
* **max_layer**: Int that specifies the maximum number of layers, default 4
* **block_threshold**: Float that specifies the threshold on the relative performance improvement when adding new block, default 1e-4. See equation (7) in `here <https://arxiv.org/pdf/1804.05093.pdf>`_ for more details.
* **layer_threshold**: Float that specifies the threshold on the relative performance improvement when evaluating new hidden layer, default 1e-4. See equation (8) in `here <https://arxiv.org/pdf/1804.05093.pdf>`_ for more details.
* **least_square_regularizer**: Float that specifies the coefficient of regulariztion when solving least square problem, default 0.1

.. _homlgop-model:

`Homogeneous Multilayer Operational Perceptron (HoMLGOP) <https://arxiv.org/pdf/1804.05093.pdf>`_
=======================================================

Description
-----------

HoMLGOP is a variant of HeMLGOP with the difference that all blocks in the same layer share the same operator set. That means the operator set searching procedure is only made when adding the 1st block of a new hidden layer. 2nd, 3rd... blocks make the same operator set assignment and only update the block weights. There is also the weights finetuning step in HoMLGOP when the architecture is defined after the progressive learning step.

.. _hemlrn-model:

`Heterogeneous Multilayer Randomized Network (HeMLRN) <https://arxiv.org/pdf/1804.05093.pdf>`_
===============================================================================================

Description
-----------

HeMLRN is also a variant of HeMLGOP with the difference is that there is no intermediate weight update steps through Back Propagation in HeMLRN. The algorithm adds a new block by using random weights and only optimize the output layer weights. After that, the random weight of the new block is fixed and another new block is learned by evaluating all operator set assignment in the same manner. All weights finetuning is also done after the network architecture is defined.

.. _homlrn-model:

`Homogeneous Multilayer Randomized Network (HoMLRN) <https://arxiv.org/pdf/1804.05093.pdf>`_
============================================================================================

Description
-----------

HoMLRN is also a variant of the above three algorithms. HoMLRN is similar to HoMLGOP in that all blocks in the same layer share the same operator set assignment. Different from HoMLGOP, HoMLRN has no intermediate weight update steps by Back Propagation. So the 2nd, 3rd... blocks of all hidden layers are assigned random weights and at each step, only the weights of output layer is optimized by solving least square problem and the performance is recorded for the new block. Whole network finetuning is also done in HoMLRN after the network growth stops. 


*Note that since HoMLGOP, HeMLRN, HoMLRN are variants of HeMLGOP, they share the same parameters as HeMLGOP described above*

.. _popfast-model:

`Fast Progressive Operational Perceptron (POPfast) <https://arxiv.org/pdf/1808.06377.pdf>`_
============================================================================================

Description
-----------

POPfast is a fast and simplified version of POP. When adding a new hidden layer, POP has to search for the operator sets of both the hidden and output layer, which involves a large search space. POPfast simply assumes a linear output layer, i.e. *multiplcation* as the nodal operator and *summation* as the pooling operator. This constraint reduces the search problem to only the new hidden layer. The progression in POPfast is similar to POP, that is the network is grown layer-wise with a predefined maximum topology. Parameters that are specific to POP are also applied to POPfast.

.. _popmem-model:

`Progressive Operational Perceptron with Memory (POPmem) <https://arxiv.org/pdf/1808.06377.pdf>`_
=================================================================================================

Description
-----------

POPmem uses a similar search procedure as in POPfast with the assumption of a linear output layer. The idea of POPmem is to augment the network growing procedure by enhancing the representation in the network. POPmem aims to address the following problem:

    *When adding a new hidden layer, POP or POPfast aims to learn a better transformation of the data by only using the output of the previous transformation (the current hidden layer), and using this (potentially better) transformation to learn a decision function (the output layer). Thus, the new hidden layer has no direct access to previously extracted hidden representations, and the output layer also has no direct access to these information*

There are two memory schemes which are denoted as POPmemH and POPmemO that was proposed to address the above problem:

* In POPmemH, before adding a new hidden layer, the previous hidden representation is linearly projected to a meaningful subspace such as PCA or LDA and concatenated to the current hidden representation as input to the new hidden layer.

* In POPmemO, when adding a new hidden layer, the *current* hidden representation is linearly projected to a meaningful subspace such as PCA or LDA. This compact representation is concanated with the new hidden layer to form an enhanced hidden representation, which is connected to the output layer. 

The motivation and discussion of two memory schemes are discussed in details in `here <https://arxiv.org/pdf/1808.06377.pdf>`_. Generally, POPmem can be understood as positing the layer addition as: given all the previously extracted hidden representations, find a new hidden layer *and* the output layer configuration that improves the performance.


Parameters
----------

The following parameters are specific to POPmem that should be set in **params** dictionary:

* **max_topology**: List of ints that specifies the maximum network topology, default [40,40,40,40], which means 4 hidden layers with 40 GOPs each
* **layer_threshold**: Float that specifies the threshold on the relative improvement, default 1e-4. See equation (8) in `here <https://arxiv.org/pdf/1804.05093.pdf>` for more details.
* **memory_type**: String that specifies the type of linear projection, either 'PCA' or 'LDA', default 'PCA'. Note that 'LDA' should be used in classification problem only. The dimension of the subspace in PCA is chosen so that at least 98% of the energy of the data is preserved. For 'LDA', the subspace dimension is 'output_dim'-1.
* **memory_regularizer**: Float that specifies the regularization coefficient when calculating the projection, default 0.1


 
