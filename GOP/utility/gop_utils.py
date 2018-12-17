#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Utility for training GOP-based models

Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

from . import gop_operators, misc
import os, glob, dill, copy, subprocess, time, numpy as np
dill.settings['recurse'] = True

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
CUDA_FLAG = 'CUDA_VISIBLE_DEVICES'
    

def he_init(fan_in, fan_out, shape):
    """Initialize layer weights using He init method
    
    Args:
        fan_in (int): Number of input neurons
        fan_out (int): Number of output neurons
        shape (list): Shape of layer weight
    
    Returns:
        Numpy array of given shape
        
    Reference:
        https://arxiv.org/abs/1502.01852
        
    """
    
    scale=1.0/max(1.0,float(fan_in+fan_out)/2)
    limit = np.sqrt(3. * scale)
    
    return np.random.normal(-limit, limit, shape).astype('float32')
    
    
def get_random_gop_weight(input_dim, 
                      output_dim, 
                      use_bias=True):
    """Initialize weights for a GOP block
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        use_bias (bool): True if using bias, False otherwise
    
    Returns:
        List of numpy arrays
        
    """
    
    W = np.random.uniform(-2*np.pi, 2*np.pi, (1, input_dim, output_dim)).astype('float32') 
    
    if use_bias:
        bias = np.random.uniform(-2*np.pi, 2*np.pi, (output_dim,)).astype('float32')
        weight = [W, bias]
    else:
        weight = [W,]
        
    return weight

def get_random_block_weights(prev_block):
    """Create GOP block with random weights with shape similar to a given block
    
    Args:
        prev_block (list): Given block weights as a reference
    
    Returns:
        List of numpy arrays
    """
    prev_gop, prev_bn, prev_output = prev_block
    
    gop_input = prev_gop[0].shape[1]
    gop_output = prev_gop[0].shape[2]
    gop = get_random_gop_weight(gop_input, gop_output, True if len(prev_gop)==2 else False)
    
    bn = [np.ones((gop_output,), dtype=np.float64), 
          np.zeros((gop_output,), dtype=np.float64), 
          np.zeros((gop_output,), dtype=np.float64),
          np.ones((gop_output,), dtype=np.float64)]
    
    hidden_dim, output_dim = prev_output[0].shape
    
    o_shape = [gop_output, output_dim]
    
    output = [np.concatenate((prev_output[0], he_init(o_shape[0], o_shape[1], o_shape)), axis=0)]
    
    if len(prev_output)==2:
        output.append(prev_output[1])
        
    return gop, bn, output


def calculate_memory_block_standalone(params,
                           train_states,
                           train_func,
                           train_data,
                           val_func,
                           val_data,
                           test_func,
                           test_data):
    
    """Standalone function to alculate memory block for POPmemH and POPmemO model
    
    Args:
        params (dict): Model parameters given as dictionary
        train_states (dict): Current topology configuration
        train_func (function): Function to generate train generator and train steps
        train_data: Input to train_func
        val_func (function): Function to generate validation generator and validation steps
        val_data: Input to val_func
        test_func (function): Function to generate test generator and test steps
        test_data: Input to test_func
        
    Returns:
        List of numpy arrays of memory block weights
    """

    misc.dump_data(params,
               train_states,
               train_func,
               train_data,
               val_func,
               val_data,
               test_func,
               test_data)
    
    runnable = 'calculate_memory.py'
    
    path = os.path.join(params['tmp_dir'], params['model_name'])
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), runnable)
    cmd = 'memory_block_path=%s python %s' %(path, filename)
    p = subprocess.Popen(cmd, shell=True)

    p.wait()

    if not os.path.exists(os.path.join(path, 'memory_block_finish.txt')):
        raise Exception('Calculate memory block finished without results!')

    with open(os.path.join(path, 'memory_block.pickle'),'rb') as fid:
        result = pickle.load(fid)

    removed_files = glob.glob(os.path.join(path, 'memory_block*'))
    misc.remove_files(removed_files)
    
    return result['pre_bn_weight'], result['projection'], result['post_bn_weight']


def calculate_memory_block(params,
                           train_states,
                           train_func,
                           train_data):
    """Calculate memory block for POPmemH and POPmemO model
    
    Args:
        params (dict): Model parameters given as dictionary
        train_states (dict): Current topology configuration
        train_func (function): Function to generate train generator and train steps
        train_data: Input to train_func
        
    Returns:
        List of numpy arrays of memory block weights
    """
    if CUDA_FLAG in os.environ.keys():
        cuda_status = os.environ[CUDA_FLAG]
    else:
        cuda_status = ''
        
    if params['finetune_computation'][0] == 'cpu':
        os.environ[CUDA_FLAG] = ''
    else:
        os.environ[CUDA_FLAG] = misc.get_gpu_str(params['search_computation'][1])
    
    try:
        if params['memory_type'] == 'PCA':
            pre_bn_weight, projection, post_bn_weight = PCA(params, train_states, train_func, train_data)
    
        elif params['memory_type'] == 'LDA':
            pre_bn_weight, projection, post_bn_weight = LDA(params, train_states, train_func, train_data)
        else:
            raise Exception('Only supported 2 memory schemes: "PCA" and "LDA", given %s' % params['memory_type'])
            
    except:
        os.environ[CUDA_FLAG] = cuda_status
        raise Exception('Failed to calculate memory block')
    
    os.environ[CUDA_FLAG] = cuda_status
    
    return pre_bn_weight, projection, post_bn_weight

def PCA(params, train_states, train_func, train_data):
    """PCA as a memory block
    
    Args:
        params (dict): Model parameters given as dictionary
        train_states (dict): Current topology configuration
        train_func (function): Function to generate train generator and train steps
        train_data: Input to train_func
        
    Returns:
        List of numpy arrays of memory block weights
    """

    from keras.models import Model
    from keras import backend as K
    
    layer_iter = train_states['layer_iter']
    input_dim = train_states['topology'][0]
    gen, steps = train_func(train_data)
    epsilon = K.epsilon()
    op_sets = misc.map_operator_from_index(train_states['op_set_indices'], params['nodal_set'], params['pool_set'], params['activation_set'])
    
    if layer_iter > 0:
            
        model = network_builder(train_states['topology'],
                                op_sets,
                                params['input_dropout'],
                                params['dropout'],
                                params['weight_regularizer'],
                                params['weight_constraint'],
                                params['output_activation'],
                                params['use_bias'])
        
        model.compile(params['optimizer'], params['loss'], params['metrics'])
        
        weights = train_states['weights']
        for layer_name in weights.keys():
            if layer_name != 'output':
                model.get_layer(layer_name).set_weights(weights[layer_name])
        
        hidden_layer_name = 'concat_' + str(layer_iter-1)
        hidden_model = Model(inputs=model.input, outputs=model.get_layer(hidden_layer_name).output)
        
    if params['direct_computation']:
        x = np.zeros((1, input_dim), dtype=np.float64)
        
        for _ in range(steps):
            x_, _ = next(gen)
            x = np.concatenate((x, x_), axis=0)
        
        x = x[1:]
        if layer_iter > 0:
            x = hidden_model.predict(x)
            
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_var = np.var(x, axis=0, keepdims=True)
        
        
        x = (x - x_mean)/(np.sqrt(x_var + epsilon))
        
        cov = np.dot(x.T, x) + params['memory_regularizer']*np.eye(x.shape[1], x.shape[1]).astype('float32')
        
    else:
        x_mean = 0.0
        N = 0
        for _ in range(steps):
            x_, _ = next(gen)
            if layer_iter > 0:
                x_ = hidden_model.predict(x_)
            x_mean += np.sum(x_, axis=0, keepdims=True) / 1e5
            N += x_.shape[0]
        
        x_mean = x_mean*1e5 / float(N)
        
        x_var = 0.0
        for _ in range(steps):
            x_, _ = next(gen)
            if layer_iter > 0:
                x_ = hidden_model.predict(x_)
            x_var += np.sum((x_ - x_mean)**2, axis=0, keepdims=True) / float(N)
            
        
        cov = 0.0
        for _ in range(steps):
            x_, _ = next(gen)
            if layer_iter > 0:
                x_ = hidden_model.predict(x_)
            x_ = (x_ - x_mean)/(np.sqrt(x_var + epsilon))
            cov += np.dot(x_.T, x_)
        
        cov += params['memory_regularizer']*np.eye(cov.shape[0])
        
    U,S,V = np.linalg.svd(cov)
    energy = np.cumsum(S)/ np.sum(S)
    idx = np.where(energy > params['min_energy_percentage'])[0][0]
    
    if idx < S.size-1:
        idx = idx +1
    
    projection = [U[:,:idx],] 
    pre_bn_weight = [np.ones((x_mean.size,), dtype=np.float64), 
                     np.zeros((x_mean.size,), dtype=np.float64), 
                     x_mean.flatten(), 
                     x_var.flatten()]
    
    bn_weight = [np.ones((idx,), dtype=np.float64), 
                 np.zeros((idx,), dtype=np.float64), 
                 np.zeros((idx,), dtype=np.float64), 
                 np.ones((idx,), dtype=np.float64)]
    
    if params['use_bias']:
        projection.append(np.zeros((idx,), dtype='float32'))
        

    K.clear_session()
    del Model
    del K
            
    return pre_bn_weight, projection, bn_weight 

def LDA(params, train_states, train_func, train_data):
    """LDA as a memory block
    
    Args:
        params (dict): Model parameters given as dictionary
        train_states (dict): Current topology configuration
        train_func (function): Function to generate train generator and train steps
        train_data: Input to train_func
        
    Returns:
        List of numpy arrays of memory block weights
    """

    from keras.models import Model
    from keras import backend as K
    
    layer_iter = train_states['layer_iter']
    input_dim = params['input_dim']
    output_dim = params['output_dim']
    gen, steps = train_func(train_data)
    epsilon = K.epsilon()
    op_sets = misc.map_operator_from_index(train_states['op_set_indices'], params['nodal_set'], params['pool_set'], params['activation_set'])
    
    if layer_iter > 0:
        
        model = network_builder(train_states['topology'],
                                op_sets,
                                params['input_dropout'],
                                params['dropout'],
                                params['weight_regularizer'],
                                params['weight_constraint'],
                                params['output_activation'],
                                params['use_bias'])
        
        model.compile(params['optimizer'], params['loss'], params['metrics'])
        
        weights = train_states['weights']
        for layer_name in weights.keys():
            if layer_name != 'output':
                model.get_layer(layer_name).set_weights(weights[layer_name])
        
        hidden_layer_name = 'concat_' + str(layer_iter-1)
        hidden_model = Model(inputs=model.input, outputs=model.get_layer(hidden_layer_name).output)
        
    if params['direct_computation']:
        x = np.zeros((1, input_dim), dtype=np.float64)
        y = np.zeros((1, output_dim), dtype=np.float64) 
        
        for _ in range(steps):
            x_, y_ = next(gen)
            x = np.concatenate((x, x_), axis=0)
            y = np.concatenate((y, y_), axis=0)
            
        x = x[1:]
        y = np.argmax(y[1:], axis=-1)
        if layer_iter > 0:
            x = hidden_model.predict(x)
            
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_var = np.var(x, axis=0, keepdims=True)
        
        
        x = (x - x_mean)/(np.sqrt(x_var + epsilon))
        
        Sb = np.zeros((x.shape[-1], x.shape[-1]), dtype=np.float64)
        Sw = np.zeros((x.shape[-1], x.shape[-1]), dtype=np.float64)
        
        for c in range(output_dim):
            class_indices = np.where(y == c)[0]
            class_mean = np.mean(x[class_indices], axis=0, keepdims=True)
            Sb += np.dot(class_mean.T, class_mean)*len(class_indices)
            
            dif = x[class_indices] - class_mean
            Sw += np.dot(dif.T, dif)
        
        Sw += params['memory_regularizer'] * np.eye(Sw.shape[0])
        
    else:
        
        x_mean = 0.0
        class_pop = [0.0,] * output_dim
        class_means = [0.0,] * output_dim
        
        for _ in range(steps):
            x_, y_ = next(gen)
            y_ = np.argmax(y_, axis=-1)
            if layer_iter > 0:
                x_ = hidden_model.predict(x_)
            x_mean += np.sum(x_, axis=0, keepdims=True) / 1e5
            
            for c in range(output_dim):
                class_indices = np.where(y_ == c)[0]
                if len(class_indices) > 0:
                    class_means[c] += np.sum(x_[class_indices], axis=0, keepdims=True) / 1e5
                    class_pop[c] += len(class_indices)
            
        Sb = np.zeros((x_mean.size, x_mean.size), dtype=np.float64)
        
        for c in range(output_dim):
            class_means[c] = class_means[c]*1e5 / float(class_pop[c])
            Sb += np.dot(class_means[c].T, class_means[c])*class_pop[c]
            
        x_mean = x_mean*1e5 / float(np.sum(class_pop))
        
        x_var = 0.0
        for _ in range(steps):
            x_, _ = next(gen)
            if layer_iter > 0:
                x_ = hidden_model.predict(x_)
            x_var += np.sum((x_ - x_mean)**2, axis=0, keepdims=True) / float(np.sum(class_pop))
            
        
        Sw = np.zeros((x_mean.size, x_mean.size), dtype=np.float64)
        
        for _ in range(steps):
            x_, y_ = next(gen)
            y_ = np.argmax(y_, axis=-1)
            if layer_iter > 0:
                x_ = hidden_model.predict(x_)
            
            x_ = (x_ - x_mean)/(np.sqrt(x_var + epsilon))
            
            for c in range(output_dim):
                class_indices = np.where(y_ == c)[0]
                if len(class_indices) > 0:
                    dif = x_[class_indices] - class_means[c]
                    Sw += np.dot(dif.T, dif)
        
        Sw += params['memory_regularizer'] * np.eye(Sw.shape[0])
        
    U,S,V = np.linalg.svd(np.dot(np.linalg.inv(Sw), Sb))
    
    projection = [U[:,:output_dim -1],] 
    pre_bn_weight = [np.ones((x_mean.size,), dtype=np.float64), 
                     np.zeros((x_mean.size,), dtype=np.float64), 
                     x_mean.flatten(), 
                     x_var.flatten()]
    
    bn_weight = [np.ones((output_dim-1,), dtype=np.float64), 
                 np.zeros((output_dim-1,), dtype=np.float64), 
                 np.zeros((output_dim-1,), dtype=np.float64), 
                 np.ones((output_dim-1,), dtype=np.float64)]
    
    if params['use_bias']:
        projection.append(np.zeros((output_dim-1,), dtype='float32'))
        

    K.clear_session()
    del Model
    del K
        
    return pre_bn_weight, projection, bn_weight

def search_cluster(params,
                    train_states,
                    train_func,
                    train_data,
                    val_func,
                    val_data,
                    test_func,
                    test_data):
    """Perform operator set search procedure on cluster
    
    Args:
        params (dict): Model parameters given as dictionary
        train_states (dict): Current topology configuration
        train_func (function): Function to generate train generator and train steps
        train_data: Input to train_func
        val_func (function): Function to generate validation generator and steps
        val_data: Input to val_func
        test_func (function): Function to generate test generator and steps
        test_data: Input to test_func
        
    Returns:
        performance (dict): Performance of best operator set at best model setting
        weights (list): Weights of new block
        op_set (tuple): Best performing operator set
        history (dict): full history (loss, metrics) 
    """
    
    misc.dump_data(params,
                   train_states,
                   train_func,
                   train_data,
                   val_func,
                   val_data,
                   test_func,
                   test_data)
    
    runnable = 'gop_search.py'
    
    path = os.path.join(params['tmp_dir'], params['model_name'])
    batch_job_filename = os.path.join(path, 'batchjob.sh')
    
    # write batch_job files if not exist
    if not os.path.exists(batch_job_filename):
        batchjob_parameters = params['batchjob_parameters']
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), runnable)
        cmd = 'gop_search_path=%s %s %s -N %d -n "$SLURM_ARRAY_TASK_ID" \n' % (path, batchjob_parameters['python_cmd'], filename, batchjob_parameters['no_machine'])
        contents = misc.setup_batchjob_file(batchjob_parameters, cmd, path)
        
        with open(batch_job_filename,'w') as fid:
            for row in contents:
                fid.write(row)
        
    # submit batch job
    cmd = 'sbatch --parsable %s' % batch_job_filename
    try:
        job_id = subprocess.check_output(cmd, shell=True)
        job_id = job_id.decode('utf-8')
    except:
        cmd = 'sbatch %s' % batch_job_filename
        try:
            output = subprocess.check_output(cmd, shell=True)
            output = output.decode('utf-8')
            job_id = output.split(' ')[-1]
        except:
            raise Exception('Failed to submit batch job')
            
    
    # loop & wait for result
    while len(glob.glob(os.path.join(path, 'machine_*_finish.txt'))) != params['batchjob_parameters']['no_machine']:
        job_status = subprocess.check_output('squeue -j %s' % job_id, shell=True)
        job_status = job_status.decode('utf-8')
        if len(job_status.split('\n'))==2:
            raise Exception('Search process using cluster terminated but no returned result exist!\n' +\
                        'Errors potentially happen during search process')
        time.sleep(5)
    
    if params['direction'] == 'higher':
        measure = np.NINF
        sign = 1.0
    else:
        measure = np.inf
        sign = -1.0
        
    machine_index = -1
    
    with open(os.path.join(path, 'machine_result.txt'), 'r') as fid:
        content = fid.read().split('\n')[:-1]

    for row in content:
        idx, p = row.split(',')
        if sign*(float(p) - measure) > 0:
            measure = float(p)
            machine_index = int(idx)
    
    with open(os.path.join(path, 'machine_%d.pickle' % machine_index), 'rb') as fid:
        result = pickle.load(fid)
        
    performance = result['performance']
    weights = result['weights']
    op_set_idx = result['op_set_idx']
    history = result['history']
    
    # clean intermediate results
    batchjob_name = params['batchjob_parameters']['name']
    removed_files =  glob.glob(os.path.join(path, 'machine_*')) +\
                     glob.glob(os.path.join(path, batchjob_name, '*.e')) + \
                     glob.glob(os.path.join(path, batchjob_name, '*.o'))
                     
    
    misc.remove_files(removed_files)
        
    return performance, weights, op_set_idx, history

def search_cpu(params,
               train_states,
               train_func,
               train_data,
               val_func,
               val_data,
               test_func,
               test_data):
    """Perform operator set search procedure on local machine using CPU
    
    Args:
        params (dict): Model parameters given as dictionary
        train_states (dict): Current topology configuration
        train_func (function): Function to generate train generator and train steps
        train_data: Input to train_func
        val_func (function): Function to generate validation generator and steps
        val_data: Input to val_func
        test_func (function): Function to generate test generator and steps
        test_data: Input to test_func
        
    Returns:
        performance (dict): Performance of best operator set at best model setting
        weights (list): Weights of new block
        op_set (tuple): Best performing operator set
        history (dict): full history (loss, metrics) 
    """
    
    misc.dump_data(params,
                  train_states,
                  train_func,
                  train_data,
                  val_func,
                  val_data,
                  test_func,
                  test_data)
    
    runnable = 'gop_search.py'
    
    path = os.path.join(params['tmp_dir'], params['model_name'])
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), runnable)
    cmd = 'CUDA_VISIBLE_DEVICES="" gop_search_path=%s python %s -N %d -n %d' %(path, filename, 1, 0)
    p = subprocess.Popen(cmd, shell=True)
    
    p.wait()
    
    if not os.path.exists(os.path.join(path, 'machine_0_finish.txt')):
        raise Exception('Search process using CPU terminated but no returned result exist!\n' +\
                        'Errors potentially happen during search process')
    
    with open(os.path.join(path, 'machine_0.pickle'),'rb') as fid:
        result = pickle.load(fid)
        
    removed_files = glob.glob(os.path.join(path, 'machine*'))
    misc.remove_files(removed_files)
    
    return result['performance'], result['weights'], result['op_set_idx'], result['history']

def search_gpu(params,
               train_states,
               train_func,
               train_data,
               val_func,
               val_data,
               test_func,
               test_data):
    """Perform operator set search procedure on local machine using GPU
    
    Args:
        params (dict): Model parameters given as dictionary
        train_states (dict): Current topology configuration
        train_func (function): Function to generate train generator and train steps
        train_data: Input to train_func
        val_func (function): Function to generate validation generator and steps
        val_data: Input to val_func
        test_func (function): Function to generate test generator and steps
        test_data: Input to test_func
        
    Returns:
        performance (dict): Performance of best operator set at best model setting
        weights (list): Weights of new block
        op_set (tuple): Best performing operator set
        history (dict): full history (loss, metrics) 
    """
    misc.dump_data(params,
               train_states,
               train_func,
               train_data,
               val_func,
               val_data,
               test_func,
               test_data)
    
    path = os.path.join(params['tmp_dir'], params['model_name'])
    no_op_set = params['no_op_set']
    performance, weights, op_set_idx, history = search_gpu_(path,
                                                       params,
                                                       0,
                                                       no_op_set,
                                                       0)
    
    
    return performance, weights, op_set_idx, history
    
    
def search_gpu_(path,
                params,
                start_idx,
                stop_idx,
                machine_no):
    """Auxiliary function to distribute search jobs on different GPUs
    
    Args:
        path (str): Path to directory holding temporary data
        params (dict): Model parameters
        start_idx (int): Start index in the list of operator sets
        stop_idx (int): Stop index in the list of operator sets
        machine_no (int): Index of machine that is given start_idx, stop_idx
        
    Returns:
        performance (dict): Performance of best operator set from start_idx to stop_idx
        weights (list): Weights of new block
        op_set (tuple): Best performing operator set
        history (dict): full history (loss, metrics) 
        
    """

    start_indices, stop_indices = misc.partition_indices(start_idx, stop_idx, len(params['search_computation'][1]))
    
    runnable = 'gop_search_gpu.py'
    
    P = []
    for gpu, b, e in zip(params['search_computation'][1], start_indices, stop_indices):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), runnable)
        cmd = 'gop_search_path=%s gop_machine_no=%d CUDA_VISIBLE_DEVICES=%d python %s -b %d -e %d' %(path,machine_no, gpu, filename, b, e)
        print(cmd)
        P.append(subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE))
        
    for p in P:
        p.wait()
        
    if len(glob.glob(os.path.join(path,'machine_%d_*_finish.txt' % machine_no))) != len(start_indices):
        raise Exception('Search process using GPU terminated but no returned result exist\n' + \
                        'Errors potentially happen during search process')
    
    with open(os.path.join(path, 'machine_%d_result.txt' % machine_no), 'r') as fid:
        content = fid.read().split('\n')[:-1]
    
    if params['direction'] == 'higher':
        measure = np.NINF
        sign = 1.0
    else:
        measure = np.inf
        sign = -1.0
        
    gpu = -1
    
    for row in content:
        idx, p = row.split(',')
        if sign*(float(p)-measure) > 0:
            measure = float(p)
            gpu = int(idx)
    
    with open(os.path.join(path, 'machine_%d_%d.pickle' % (machine_no, gpu)), 'rb') as fid:
        irsgpu_result = pickle.load(fid)
        
    performance = irsgpu_result['performance']
    weights = irsgpu_result['weights']
    op_set_idx = irsgpu_result['op_set_idx']
    history = irsgpu_result['history']
    
    removed_files = glob.glob(os.path.join(path, 'machine_*'))
    misc.remove_files(removed_files)
    
    return performance, weights, op_set_idx, history
    

def GIS(start_idx,
        stop_idx,
        params,
        train_states):
    """Greedy Iterative Search for POP
    
    Perform one iterative search pass for POP algorithm
    
    Args:
        start_idx (int): Start index in the list of operator sets
        stop_idx (int): Stop index in the list of operator sets
        params (dict): Model parameters
        train_states (dict): Current topology configuration
        
    Returns:
        performance (dict): Performance of best operator set from start_idx to stop_idx
        weights (list): Weights of new block
        op_set (tuple): Best performing operator set
        history (dict): full history (loss, metrics) 
        
    """

    from joblib import Parallel, delayed
        
    
    op_set_indices = range(params['no_op_set'])
    
    if start_idx is not None:
        all_op_set_indices = op_set_indices[start_idx:stop_idx]
        
    if params['search_computation'][0] == 'cpu':
        no_process = params['search_computation'][1]
    else:
        no_process = 1
    
    search_results = Parallel(n_jobs=no_process, temp_folder=os.path.join(params['tmp_dir'], params['model_name']))\
            (delayed(GIS_)(params, train_states, op_set_idx) for op_set_idx in all_op_set_indices)
    
    performance, weights, op_set_idx, history = get_optimal_op_set(search_results, params['convergence_measure'], params['direction'])
    
    return performance, weights, op_set_idx, history


    return

def GIS_(params,
         train_states,
         op_set_idx):
    """Auxiliary function of GIS to evaluate the given operator set
    
    Args:
        params (dict): Model parameters
        train_states (dict): Current topology setting
        op_set_idx (int): Operator set index for new block
    
    Returns:
        block_performance (dict): Performance when adding new block
        weights (list): Weights of new block
        op_set_idx (int): The given op_set index
        history (dict): full history (loss, metrics) 
    """
    
    path = os.path.join(params['tmp_dir'], params['model_name'])
    train_func, train_data, val_func, val_data, test_func, test_data = misc.unpickle_generator(os.path.join(path, 'data.pickle'))
    
    if os.path.exists(os.path.join(path, 'custom_loss.pickle')):
        params['loss'] = misc.unpickle_custom_loss(os.path.join(path, 'custom_loss.pickle'))
    
    if os.path.exists(os.path.join(path, 'custom_metrics.pickle')):
        params['metrics'] = misc.unpickle_custom_metrics(os.path.join(path, 'custom_metrics.pickle'))
        
    if os.path.exists(os.path.join(path, 'special_metrics.pickle')):
        params['special_metrics'] = misc.unpickle_special_metrics(os.path.join(path, 'special_metrics.pickle'))
        
    if os.path.exists(os.path.join(path, 'custom_operators.pickle')):
        params['nodal_set'], params['pool_set'], params['activation_set'] = misc.unpickle_custom_operators(os.path.join(path, 'custom_operators.pickle'))
    
    op_set_indices = copy.deepcopy(train_states['op_set_indices'])
    weights = copy.deepcopy(train_states['weights'])
    layer_iter = train_states['layer_iter']
    
    suffix = '_' + str(layer_iter) + '_0' 
    
    if train_states['search_layer'] == 'output':
        op_set_indices['output'] = op_set_idx
        op_set_indices['gop' + suffix] = train_states['hidden_op_set_idx']
    else:
        op_set_indices['gop' + suffix] = op_set_idx
        op_set_indices['output'] = train_states['output_op_set_idx']
    
    prev_hidden_dim = params['input_dim'] if layer_iter == 0 else params['max_topology'][layer_iter-1]
    block_size = params['max_topology'][layer_iter]
    
    weights['gop' + suffix] = get_random_gop_weight(prev_hidden_dim, block_size, params['use_bias'])
    weights['bn' + suffix] = [np.ones((block_size,), dtype=np.float64),
                              np.zeros((block_size,), dtype=np.float64),
                              np.zeros((block_size,), dtype=np.float64),
                              np.ones((block_size,), dtype=np.float64),]
    weights['output'] = get_random_gop_weight(params['max_topology'][layer_iter], params['output_dim'], params['use_bias'])        
    
    block_names = ['gop'+suffix, 'bn'+suffix, 'output']
    
    
    
    
    measure, history, weights = block_update(train_states['topology'], 
                                             op_set_indices, 
                                             weights, 
                                             params, 
                                             block_names, 
                                             train_func,
                                             train_data,
                                             val_func,
                                             val_data,
                                             test_func,
                                             test_data)
    
    
    block_weights = [weights[block_names[0]], weights[block_names[1]], weights[block_names[2]]]
    
    idx = np.argmax(history[params['convergence_measure']]) if params['direction'] == 'higher' else np.argmin(history[params['convergence_measure']])
    block_performance = {}
    
    for metric in history.keys():
        if history[metric] is not None:
            block_performance[metric] = history[metric][idx]
        
    
    return block_performance, block_weights, op_set_idx, history


def GISfast(start_idx,
            stop_idx,
            params, 
            train_states):
    """Fast version of Greedy Iterative Search
    
    This search procedure assumes the output layer is a linear layer
    
    Args:
        start_idx (int): Start index in the list of operator sets
        stop_idx (int): Stop index in the list of operator sets
        params (dict): Model parameters
        train_states (dict): Current topology configuration
        
    Returns:
        performance (dict): Performance of best operator set from start_idx to stop_idx
        weights (list): Weights of new block
        op_set (tuple): Best performing operator set
        history (dict): full history (loss, metrics) 
    
    """
    
    from joblib import Parallel, delayed
        
    
    all_op_set_indices = range(params['no_op_set'])
    
    if start_idx is not None:
        all_op_set_indices = all_op_set_indices[start_idx:stop_idx]
        
    if params['search_computation'][0] == 'cpu':
        no_process = params['search_computation'][1]
    else:
        no_process = 1
    
    search_results = Parallel(n_jobs=no_process, temp_folder=os.path.join(params['tmp_dir'], params['model_name']))\
            (delayed(GISfast_)(params, train_states, op_set_idx) for op_set_idx in all_op_set_indices)
    
    performance, weights, op_set_idx, history = get_optimal_op_set(search_results, params['convergence_measure'], params['direction'])
    
    return performance, weights, op_set_idx, history


def GISfast_(params,
             train_states,
             op_set_idx):
    """Auxiliary function of GISfast to evaluate the given operator set
    
    Args:
        params (dict): Model parameters
        train_states (dict): Current topology setting
        op_set_idx (int): Operator set index for new block
    
    Returns:
        block_performance (dict): Performance when adding new block
        weights (list): Weights of new block
        op_set_idx (int): The given op_set_idx
        history (dict): full history (loss, metrics) 
    """
    
    path = os.path.join(params['tmp_dir'], params['model_name'])
    train_func, train_data, val_func, val_data, test_func, test_data = misc.unpickle_generator(os.path.join(path, 'data.pickle'))
    
    if os.path.exists(os.path.join(path, 'custom_loss.pickle')):
        params['loss'] = misc.unpickle_custom_loss(os.path.join(path, 'custom_loss.pickle'))
    
    if os.path.exists(os.path.join(path, 'custom_metrics.pickle')):
        params['metrics'] = misc.unpickle_custom_metrics(os.path.join(path, 'custom_metrics.pickle'))
        
    if os.path.exists(os.path.join(path, 'special_metrics.pickle')):
        params['special_metrics'] = misc.unpickle_special_metrics(os.path.join(path, 'special_metrics.pickle'))
        
    if os.path.exists(os.path.join(path, 'custom_operators.pickle')):
        params['nodal_set'], params['pool_set'], params['activation_set'] = misc.unpickle_custom_operators(os.path.join(path, 'custom_operators.pickle'))
        
    layer_iter = train_states['layer_iter']
    block_iter = train_states['block_iter']
    block_suffix = '_' + str(layer_iter) + '_' + str(block_iter)
    block_size = params['max_topology'][layer_iter]
    
    if layer_iter == 0:
        prev_hidden_dim = train_states['topology'][0]
    else:
        prev_hidden_dim = 0
        for block in train_states['topology'][-3]:
            prev_hidden_dim += block[1]
    
    # add new gop block
    topology = copy.deepcopy(train_states['topology'])
    topology[-2].append(('gop', block_size))
    
    hidden_dim = 0
    for block in topology[-2]:
        hidden_dim += block[1]
        
    output_dim = topology[-1][1]
    
    op_set_indices = copy.deepcopy(train_states['op_set_indices'])
    op_set_indices['gop' + block_suffix] = op_set_idx
    
    
    
    weights = copy.deepcopy(train_states['weights'])
    gop_weight = [he_init(prev_hidden_dim, block_size, (1, prev_hidden_dim, block_size)),]
    
    bn_weight = [np.ones((block_size,), dtype=np.float64), 
                 np.zeros((block_size,), dtype=np.float64), 
                 np.zeros((block_size,), dtype=np.float64),
                 np.ones((block_size,), dtype=np.float64)]
    
    output_weight = [he_init(hidden_dim, output_dim, [hidden_dim, output_dim])]
    
    if params['use_bias']:
        gop_weight.append(np.zeros((block_size,), dtype=np.float64))
        output_weight.append(np.zeros((output_dim,), dtype=np.float64))
        
    weights['gop' + block_suffix] = gop_weight
    weights['bn' + block_suffix] = bn_weight
    weights['output'] = output_weight
        
    block_names = ['gop' + block_suffix,
                   'bn' + block_suffix,
                   'output']

    measure, history, weights = block_update(topology, 
                                             op_set_indices, 
                                             weights, 
                                             params, 
                                             block_names, 
                                             train_func,
                                             train_data,
                                             val_func,
                                             val_data,
                                             test_func,
                                             test_data)
    
    
    block_weights = [weights[block_names[0]], weights[block_names[1]], weights[block_names[2]]]
    
    idx = np.argmax(history[params['convergence_measure']]) if params['direction'] == 'higher' else np.argmin(history[params['convergence_measure']])
    block_performance = {}
    
    for metric in history.keys():
        if history[metric] is not None:
            block_performance[metric] = history[metric][idx]
        
    
    return block_performance, block_weights, op_set_idx, history

def IRS(start_idx,
        stop_idx,
        params, 
        train_states):
    """Iterative Randomized Search for HeMLGOP, HoMLGOP, HeMLRN, HoMLRN
    
    This function performs the evaluation of a set of operator sets using
    randomized method
    
    Args:
        start_idx (int): Start index in the list of operator sets
        stop_idx (int): Stop index in the list of operator sets
        params (dict): Model parameters
        train_states (dict): Current topology configuration
        
    Returns:
        performance (dict): Performance of best operator set from start_idx to stop_idx
        weights (list): Weights of new block
        op_set_idx (int): Index of best performing operator set
        history (dict): full history (loss, metrics) 
    
    """
    
    from joblib import Parallel, delayed
    
    op_set_indices = range(params['no_op_set'])
    
    if start_idx is not None:
        all_op_set_indices = op_set_indices[start_idx:stop_idx]
        
    if params['search_computation'][0] == 'cpu':
        no_process = params['search_computation'][1]
    else:
        no_process = 1
    
    search_results = Parallel(n_jobs=no_process, temp_folder=os.path.join(params['tmp_dir'], params['model_name']))\
            (delayed(LS)(params, train_states, op_set_idx) for op_set_idx in all_op_set_indices)
    
    performance, weights, op_set_idx, history = get_optimal_op_set(search_results, params['convergence_measure'], params['direction'])
    
    return performance, weights, op_set_idx, history



def LS(params, 
       train_states, 
       op_set_idx):
    """Auxiliary function of IRS to evaluate the given operator set
    
    Args:
        params (dict): Model parameters
        train_states (dict): Current topology setting
        op_set (int): Index of operator set of new block
    
    Returns:
        block_performance (dict): Performance when adding new block
        weights (list): Weights of new block
        op_set_idx (int): Index of the given operator set
        history (dict): full history (loss, metrics) 
    """
            
    from keras.models import Model
    from keras import backend as K
    
    path = os.path.join(params['tmp_dir'], params['model_name'])
    train_func, train_data, val_func, val_data, test_func, test_data = misc.unpickle_generator(os.path.join(path, 'data.pickle'))
    
    if os.path.exists(os.path.join(path, 'custom_loss.pickle')):
        params['loss'] = misc.unpickle_custom_loss(os.path.join(path, 'custom_loss.pickle'))
    
    if os.path.exists(os.path.join(path, 'custom_metrics.pickle')):
        params['metrics'] = misc.unpickle_custom_metrics(os.path.join(path, 'custom_metrics.pickle'))
        
    if os.path.exists(os.path.join(path, 'special_metrics.pickle')):
        params['special_metrics'] = misc.unpickle_special_metrics(os.path.join(path, 'special_metrics.pickle'))
        
    if os.path.exists(os.path.join(path, 'custom_operators.pickle')):
        params['nodal_set'], params['pool_set'], params['activation_set'] = misc.unpickle_custom_operators(os.path.join(path, 'custom_operators.pickle'))
    
    
    train_gen, train_steps = train_func(train_data)
    
    if val_func:
        val_gen, val_steps = val_func(val_data)
    else:
        val_gen, val_steps = None, None
        
    if test_func:
        test_gen, test_steps = test_func(test_data)
    else:
        test_gen, test_steps = None, None
        
    # get parameters
    layer_iter = train_states['layer_iter']
    block_iter = train_states['block_iter']
    use_bias = params['use_bias']
    block_suffix = '_' + str(layer_iter) + '_' + str(block_iter)
    block_size = params['block_size']
    weights = train_states['weights']
        
    # add new gop block
    topology = copy.deepcopy(train_states['topology'])
    topology[-2].append(('gop', block_size))
    op_set_indices = copy.deepcopy(train_states['op_set_indices'])
    op_set_indices['gop' + block_suffix] = op_set_idx
    op_sets = misc.map_operator_from_index(op_set_indices, params['nodal_set'], params['pool_set'], params['activation_set'])
    
    model = network_builder(topology,
                            op_sets,
                            params['input_dropout'],
                            params['dropout'],
                            params['weight_regularizer'],
                            params['weight_constraint'],
                            params['output_activation'],
                            use_bias)
    
    
    model.compile(params['optimizer'], loss=params['loss'], metrics=params['metrics'])
    
    # set fixed weights & random weights
    block_input_dim = 0
    if layer_iter == 0:
        block_input_dim = topology[0]
    else:
        for _, units in topology[-3]:
            block_input_dim += units
    
    last_hidden_dim = 0
    for _, units in topology[-2]:
        last_hidden_dim += units
        
    # set random weights for new block
    gop_weight = get_random_gop_weight(block_input_dim, block_size, use_bias)
    model.get_layer('gop' + block_suffix).set_weights(gop_weight)
    
    # set fixed weights
    for layer_name in train_states['weights'].keys():
        if layer_name != 'output':
            model.get_layer(layer_name).set_weights(weights[layer_name])
    
    hidden_layer_name = 'concat_' + str(layer_iter) if block_iter > 0 else 'bn' + block_suffix
    hidden_model = Model(inputs=model.input, outputs=model.get_layer(hidden_layer_name).output)
    
    bn_weight, output_weight = least_square(hidden_model, 
                                            topology[0], 
                                            last_hidden_dim, 
                                            block_size,
                                            topology[-1][-1], 
                                            train_gen, 
                                            train_steps, 
                                            params['direct_computation'], 
                                            params['least_square_regularizer'], 
                                            use_bias,
                                            K.epsilon(),
                                            params['class_weight'])
    
    
    model.get_layer('bn' + block_suffix).set_weights(bn_weight)
    model.get_layer('output').set_weights(output_weight)
    
    block_performance = misc.evaluate(model, train_gen, train_steps, val_gen, val_steps, test_gen, test_steps)
    
    if params['special_metrics']:
        train_performance = misc.evaluate_special_metrics(model, params['special_metrics'], train_gen, train_steps)
        val_performance = misc.evaluate_special_metrics(model, params['special_metrics'], val_gen, val_steps) if val_gen else None
        test_performance = misc.evaluate_special_metrics(model, params['special_metrics'], test_gen, test_steps) if test_gen else None
        
        for idx, metric in enumerate(params['special_metrics']):
            block_performance['train_' + metric.__name__] = train_performance[idx]
            block_performance['val_' + metric.__name__] = val_performance[idx] if val_performance else None
            block_performance['test_' + metric.__name__] = test_performance[idx] if test_performance else None
    
    block_weights = [gop_weight, bn_weight, output_weight]
    
    K.clear_session()
    del Model
    del K
    return block_performance, block_weights, op_set_idx, block_performance 

def least_square(model, 
                 input_dim, 
                 hidden_dim, 
                 block_size,
                 output_dim, 
                 gen, 
                 steps, 
                 direct_computation, 
                 regularizer, 
                 use_bias,
                 epsilon,
                 class_weight=None):
    """Auxiliary function of LS to solve least-square problem
    
    Args:
        model: Keras model that produce hidden representation
        input_dim (int): Dimension of input data
        hidden_dim (int): Dimension of current hidden layer
        block_size (int): Size of new block
        output_dim (int): Dimension of target
        gen (generator): Data generator
        steps (int): Number of mini-batches
        direction_computation (bool): Compute using full-batch if True, otherwise mini-batch
        regularizer (float): Regularization of least-square problem
        use_bias (bool): append bias if True
        epsilon (float): small amount added to denominator to avoid zero division
        class_weight (dict): weights to rebalance the contribution of each class to the loss function
        
    Returns:
        bn_weight (list): weights of BN layer in the block
        output_weight (list): weights of output layer
        
    """
    
    if direct_computation:
        x = np.zeros((1, input_dim), dtype=np.float64)
        y = np.zeros((1, output_dim), dtype=np.float64)
        
        for _ in range(steps):
            x_, y_ = next(gen)
            x = np.concatenate((x, x_), axis=0)
            y = np.concatenate((y, y_), axis=0)
        
        y = y[1:]
        x = model.predict(x[1:])
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_var = np.var(x, axis=0, keepdims=True)
        
        # set all except last block to zero, one 
        x_mean[:,:hidden_dim-block_size] = 0.0
        x_var[:,:hidden_dim-block_size] = 1.0 - epsilon
        
        x = (x - x_mean)/(np.sqrt(x_var + epsilon))
        
        if class_weight is not None:
            weight = np.zeros((x.shape[0],1))
            for key in class_weight.keys():
                indices = np.where(y==key)[0]
                weight[indices] = class_weight[key]
            
            x = x*weight
            y = y*weight
        
        xTx = np.dot(x.T, x)
        xTy = np.dot(x.T, y)
        W = np.dot(np.linalg.pinv(xTx + regularizer*np.eye(hidden_dim, hidden_dim, dtype=np.float64)), xTy)
        
        
    else:
        x_mean = 0.0
        N = 0
        for _ in range(steps):
            x_, _ = next(gen)
            x_ = model.predict(x_)
            x_mean += np.sum(x_, axis=0, keepdims=True) / 1e5
            N += x_.shape[0]
        
        x_mean = x_mean*1e5 / float(N)
        
        x_var = 0.0
        for _ in range(steps):
            x_, _ = next(gen)
            x_ = model.predict(x_)
            x_var += np.sum((x_ - x_mean)**2, axis=0, keepdims=True) / float(N)
            
        # set all except last block to zero, one
        x_mean[:,:hidden_dim-block_size] = 0.0
        x_var[:,:hidden_dim-block_size] = 1.0 - epsilon
        
        xTx = 0.0
        xTy = 0.0
        for _ in range(steps):
            x_, y_ = next(gen)
            x_ = model.predict(x_)
            x_ = (x_ - x_mean)/(np.sqrt(x_var + epsilon))
            
            if class_weight is not None:
                weight = np.zeros((x_.shape[0],1))
                for key in class_weight.keys():
                    indices = np.where(y_==key)[0]
                    weight[indices] = class_weight[key]
                
                x_ = x_*weight
                y_ = y_*weight
                
            xTx += np.dot(x_.T, x_)
            xTy += np.dot(x_.T, y_)
    
        W = np.dot(np.linalg.pinv(xTx + regularizer*np.eye(hidden_dim, hidden_dim, dtype=np.float64)), xTy)
        
    
    
    if use_bias:
        output_weight = [W, np.zeros((output_dim,), dtype=np.float64)]
    else:
        output_weight = [W,]
        
    
    bn_weight = [np.ones((block_size,), dtype=np.float64), 
                 np.zeros((block_size,), dtype=np.float64), 
                 x_mean.flatten()[hidden_dim-block_size:], 
                 x_var.flatten()[hidden_dim-block_size:]]
    

    
    return bn_weight, output_weight


def block_update_standalone(train_states,
                            params,
                            block_names,
                            train_func,
                            train_data,
                            val_func,
                            val_data,
                            test_func,
                            test_data):
    
    train_states['block_names'] = block_names
    
    misc.dump_data(params,
               train_states,
               train_func,
               train_data,
               val_func,
               val_data,
               test_func,
               test_data)
    
    runnable = 'block_update.py'
    
    path = os.path.join(params['tmp_dir'], params['model_name'])
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), runnable)
    cmd = 'block_update_path=%s python %s' %(path, filename)
    p = subprocess.Popen(cmd, shell=True)
    
    p.wait()
    
    if not os.path.exists(os.path.join(path, 'block_update_finish.txt')):
        raise Exception('Block update finished without results!')
    
    with open(os.path.join(path, 'block_update_output.pickle'),'rb') as fid:
        result = pickle.load(fid)
        
    removed_files = glob.glob(os.path.join(path, 'block_update*'))
    misc.remove_files(removed_files)
    
    return result['measure'], result['history'], result['weights']
    
    
def block_update(topology, 
                 op_set_indices, 
                 weights, 
                 params, 
                 block_names, 
                 train_func,
                 train_data,
                 val_func,
                 val_data,
                 test_func,
                 test_data):
    """Construct the network and update the given blocks
    
    Args:
        topology (list): Data about the network topology
        op_set_indices (dict): Operator set indices of each block
        weights (dict): Weights of each block
        params (dict): Model parameters
        block_names (list): List of layers to update
        train_func (function): Function to return train generator and #mini-batches
        train_data: Input to train_func
        val_func (function): Function to return val generator and #mini-batches
        val_data: Input to val_func
        test_func (function): Function to return test generator and #mini-batches
        test_data: Input to test_func
        
    Returns:
        measure (float): Performance according to convergence_measure
        history (dict): History (loss, metrics) of all epochs
        weights (dict): Weights after update
        
    """
    
    op_sets = misc.map_operator_from_index(op_set_indices, params['nodal_set'], params['pool_set'], params['activation_set'])
    
    model = network_builder(topology,
                            op_sets,
                            params['input_dropout'],
                            params['dropout'],
                            params['weight_regularizer'],
                            params['weight_constraint'], 
                            params['output_activation'],
                            params['use_bias'])
                
    model.compile(params['optimizer'], params['loss'], params['metrics'])
        
    model = misc.set_model_weights(model, weights)
    
    
    for layer in model.layers:
        if layer.name in block_names:
            layer.trainable = True
        else:
            layer.trainable = False
    
    measure, history, new_weights = network_trainer(model,
                                                    params['direction'],
                                                    params['convergence_measure'],
                                                    params['lr_train'], 
                                                    params['epoch_train'],
                                                    params['optimizer'],
                                                    params['loss'],
                                                    params['metrics'],
                                                    params['special_metrics'],
                                                    train_func,
                                                    train_data,
                                                    val_func,
                                                    val_data,
                                                    test_func,
                                                    test_data,
                                                    params['class_weight'])
                                    
    # get block weights
    model.set_weights(new_weights)
    for layer_name in weights.keys():
        weights[layer_name] = model.get_layer(layer_name).get_weights()
        
    from keras import backend as K
    K.clear_session()
    del K
    
    return measure, history, weights
    
    
def network_builder(topology,
                    op_sets,
                    input_dropout=None,
                    dropout=None,
                    regularizer=None,
                    constraint=None,
                    output_activation=None,
                    use_bias=True):
    """Building Keras model using given topology and parameters
    
    Args:
        topology (list): Data about network topology
        op_sets (dict): Operator sets of each block
        input_dropout (float): Dropout applied to input layer, default None
        dropout (float): Dropout applied to hidden layers, default None
        regularizer (float): Weight decay coefficient, default None
        constraint (float): Max-norm constraint of weights, default None
        output_activation (str): Optional activation applied to output layer, default None
        use_bias (bool): Allow using bias if True
        
    Returns:
        Keras model without compilation
        
    """
    
    from ..layers import gop
    from keras.layers import Dropout, Input, Dense, BatchNormalization as BN, Concatenate
    from keras.models import Model
    from keras import regularizers, constraints
    
    inputs = Input((topology[0],))
    
    if input_dropout:
        hiddens = Dropout(input_dropout, name='input_dropout')(inputs)
    else:
        hiddens = inputs
    
    no_layer = len(topology)
    
    for layer_idx in range(1,no_layer-1):
        layer_blocks = []
        layer_output_dim = 0
        no_block = len(topology[layer_idx])
        
        for block_idx in range(no_block):
            # if last block, use given op_set and block_size
            suffix = '_' + str(layer_idx-1) + '_' + str(block_idx)
            block_type, units = topology[layer_idx][block_idx]
            
            layer_output_dim += units
            
            if block_type == 'gop':
                hiddens_ = gop.GOP(units,
                                   op_sets[block_type + suffix],
                                   regularizer,
                                   constraint,
                                   use_bias,
                                   name='gop' + suffix)(hiddens)
                
                hiddens_ = BN(name = 'bn' + suffix)(hiddens_)
            
            elif block_type == 'mem':
                hiddens_ = BN(name='mem_pre_bn' + suffix)(hiddens)
                hiddens_ = Dense(units, use_bias=use_bias, name='mem'+suffix)(hiddens_)
                hiddens_ = BN(name='bn'+suffix)(hiddens_)
            
            else:
                raise 'Invalid block type ´´%s´´' % block_type
                
                
            layer_blocks.append(hiddens_)
            
        
        if len(layer_blocks) > 1:
            hiddens = Concatenate(axis=-1, name='concat_' + str(layer_idx-1))(layer_blocks)
        else:
            hiddens = layer_blocks[0]
        
        if dropout:
            hiddens = Dropout(dropout, name='dropout_' + str(layer_idx-1))(hiddens)
            
    output_type, output_units = topology[-1]
    
    if output_type == 'dense':
        outputs = Dense(output_units, 
                        kernel_regularizer=regularizers.l2(regularizer) if regularizer else None,
                        kernel_constraint=constraints.max_norm(constraint, axis=0) if constraint else None,
                        activation=output_activation, 
                        use_bias=use_bias, 
                        name='output')(hiddens)
    
    elif output_type == 'gop':
        outputs = gop.GOP(output_units, 
                          op_sets['output'], 
                          regularizer,
                          constraint,
                          use_bias,
                          name='output')(hiddens)
    
    model = Model(inputs=inputs, outputs=outputs)

    return model

        

def network_trainer(model,
                    direction,
                    convergence_measure,
                    LR, 
                    SC,
                    optimizer,
                    loss,
                    metrics,
                    special_metrics,
                    train_func,
                    train_data,
                    val_func,
                    val_data,
                    test_func,
                    test_data,
                    class_weight):
    
    """Train the given model with given parameters
    
    Args:
        model: Keras model instance
        direction (str): String indicates quality of convergence measure ('higher'/'lower')
        convergence_measure (str): Name of the metric to evaluate when to stop
        LR (list): List of learning rate schedule
        SC (list): Number of epochs for each learning rate value
        optimizer (str): Name of the optimizer
        loss (str/callable): Loss function
        metrics (list): List of metrics to evaluate
        special_metrics (list): List of special metrics to evaluate
        train_func (function): Function to return train generator and #mini-batches
        train_data: Input to train_func
        val_func (function): Function to return val generator and #mini-batches
        val_data: Input to val_func
        test_func (function): Function to return test generator and #mini-batches
        test_data: Input to test_func
        class_weight: (dict) Weight assigned to each class to rebalance the contribution of each class to the loss
        
    Returns:
        measure: The value of convergence_measure
        history: Entire history (loss/metrics) of given data
        optimal_weights: Network weights after training
        
    """
    
    # set up generator
    train_gen, train_steps = train_func(train_data)
    val_gen, val_steps = val_func(val_data) if val_func else (None, None)
        
    # set up parameters
    
    current_weights = model.get_weights()
    optimal_weights = model.get_weights()
    trainable_status = {}
    for layer in model.layers:
        trainable_status[layer.name] = layer.trainable
    
    if direction == 'higher':
        measure = np.NINF
        sign = 1.0
    else:
        measure = np.inf
        sign = -1.0
        
    history = {}
    
    for metric in model.metrics_names:
        history['train_' + metric] = []
        history['val_' + metric] = [] if val_gen else None
        history['test_' + metric] = [] if test_func else None
    
    if special_metrics:
        for metric in special_metrics:
            history['train_' + metric.__name__] = []
            history['val_' + metric.__name__] = [] if val_gen else None
            history['test_' + metric.__name__] = [] if test_func else None
    
    for lr, sc in zip(LR,SC):
        model.compile(optimizer, loss, metrics)
        model.set_weights(current_weights)
        for layer in model.layers:
            layer.trainable = trainable_status[layer.name]
        
        cb = misc.BestModelLogging(convergence_measure, 
                                   direction,
                                   special_metrics,
                                   train_func,
                                   train_data,
                                   val_func,
                                   val_data,
                                   test_func,
                                   test_data)
        #model.summary()
        model.fit_generator(train_gen,
                            train_steps,
                            epochs=sc,
                            verbose=0,
                            callbacks=[cb,],
                            validation_data=val_gen,
                            validation_steps=val_steps,
                            class_weight=class_weight)
        
        if sign*(cb.measure - measure) > 0:
            optimal_weights = cb.model_weights
            measure = cb.measure
        
        current_weights = model.get_weights()
        
        for metric in cb.performance.keys():
            if cb.performance[metric] is not None:
                history[metric] += cb.performance[metric]
    
    
    return measure, history, optimal_weights
        
def get_optimal_op_set(search_results, convergence_measure, direction):
    """Select the optimal operator set based on search results
    
    Args:
        search_results (list): List of search results of all operator sets
        convergence_measure (str): Name of the metric to decide optimality
        direction (str): String indicates relative quality of convergence_measure ('higher'/'lower')
    
    Returns:
        performance (dict): Performances of all metrics of optimal operator set
        weights (dict): Corresponding weights of the optimal operator set
        op_set (tuple): Optimal operator set
        history (dict): Full history when evaluting optimal operator set
        
        
    """
    measure = np.NINF if direction == 'higher' else np.inf
    sign = 1.0 if direction == 'higher' else -1.0
    performance = None
    weights = None
    op_set_idx = None
    history = None
    
    for r in search_results:
        if sign*(r[0][convergence_measure]-measure) > 0:
            measure = r[0][convergence_measure]
            performance = r[0]
            weights = r[1]
            op_set_idx = r[2]
            history = r[3]
    
    return performance, weights, op_set_idx, history


def finetune(model_data, 
             params,
             train_func, 
             train_data,
             val_func=None, 
             val_data=None,
             test_func=None,
             test_data=None):
    """Construct and finetune the given network
    
    Args:
        model_data (dict): Data of the model
        params (dict): hyper-parameters to finetune the model
        train_func (function): Function to return train generator and #mini-batches
        train_data: Input to train_func
        val_func (function): Function to return val generator and #mini-batches
        val_data: Input to val_func
        test_func (function): Function to return test generator and #mini-batches
        test_data: Input to test_func
        
    Returns:
        history (dict): Full history during finetuning
        performance (dict): Performance of all metrics at the best model setting
        model_data (dict): Updated model data if performance improves after finetuning
        
    """
    
    
    original_convergence_measure = params['convergence_measure']        
    if val_func:
        params['convergence_measure'] = 'val_' + params['convergence_measure'] 
    else:
        params['convergence_measure'] = 'train_' + params['convergence_measure']
    
    if model_data is None:
        raise Exception('No model data exists for finetuning \n Try load(filename) to load a trained model \n or train a new model with fit() or progressive_learn()')
        
    if params['input_dim'] != model_data['topology'][0]:
        raise Exception('given data has input dimension ´´%d´´ but existing model has input dimension ´´%d´´' %(params['input_dim'], model_data['topology'][0]))
    
    if params['output_dim'] != model_data['topology'][-1][1]:
        raise Exception('given data has output dimension ´´%d´´ but existing model has output dimension ´´%d´´' %(params['output_dim'], model_data['topology'][-1]))
    
    misc.test_generator(train_func, train_data, params['input_dim'], params['output_dim'])
    if val_func:    misc.test_generator(val_func, val_data, params['input_dim'], params['output_dim'])
    if test_func:   misc.test_generator(test_func, test_data, params['input_dim'], params['output_dim'])
    

    model = network_builder(model_data['topology'],
                                      model_data['op_sets'],
                                      params['input_dropout'],
                                      params['dropout_finetune'],
                                      params['weight_regularizer_finetune'],
                                      params['weight_constraint_finetune'],
                                      model_data['output_activation'],
                                      model_data['use_bias'])
    
    model.compile(params['optimizer'], params['loss'], params['metrics'])
        
    model = misc.set_model_weights(model, model_data['weights'])
        
    train_gen, train_steps = train_func(train_data)
    val_gen, val_steps = val_func(val_data) if val_func else (None, None)
    test_gen, test_steps = test_func(test_data) if test_func else (None, None)
        
    performance = misc.evaluate(model,
                                train_gen,
                                train_steps,
                                val_gen,
                                val_steps,
                                test_gen,
                                test_steps)
    
    if params['special_metrics']:
        train_performance = misc.evaluate_special_metrics(model, params['special_metrics'], train_gen, train_steps)
        val_performance = misc.evaluate_special_metrics(model, params['special_metrics'], val_gen, val_steps) if val_gen else None
        test_performance = misc.evaluate_special_metrics(model, params['special_metrics'], test_gen, test_steps) if test_gen else None
        
        for idx, metric in enumerate(params['special_metrics']):
            performance['train_' + metric.__name__] = train_performance[idx]
            performance['val_' + metric.__name__] = val_performance[idx] if val_performance else None
            performance['test_' + metric.__name__] = test_performance[idx] if test_performance else None
    
    measure, history, weights = network_trainer(model, 
                                                params['direction'], 
                                                params['convergence_measure'], 
                                                params['lr_finetune'], 
                                                params['epoch_finetune'], 
                                                params['optimizer'], 
                                                params['loss'], 
                                                params['metrics'], 
                                                params['special_metrics'],
                                                train_func, 
                                                train_data, 
                                                val_func, 
                                                val_data, 
                                                test_func, 
                                                test_data,
                                                params['class_weight'])
    
    sign = 1.0 if params['direction']=='higher' else -1.0
    
    if sign*(measure - performance[params['convergence_measure']]) > 0:
        model.set_weights(weights)
        
        for layer_name in model_data['weights'].keys():
            model_data['weights'][layer_name] = model.get_layer(layer_name).get_weights()
        
        idx = np.argmax(history[params['convergence_measure']]) if params['direction'] == 'higher' else np.argmin(history[params['convergence_measure']])

        for metric in performance.keys():
            if performance[metric] is not None:
                performance[metric] = history[metric][idx]
                
        
    params['convergence_measure'] = original_convergence_measure
    
    from keras import backend as K
    K.clear_session()
    del K
    
    return history, performance, model_data
    
def evaluate(model_data, func, data, metrics, special_metrics=None):
    """Construct and evaluate the network with given model configuration and data
    
    Args:
        model_data (dict): Data of the model
        func (callable): Function to generate data generator and #mini-batch
        data: Input to func
        metrics (list): List of metrics to evaluate
        special_metrics (list): List of special metrics to evaluate
        
    Returns:
        performance (dict): Performance of given metrics
        
    """
    
    if model_data is None:
        raise Exception('No model exist for evaluation')
    
    misc.test_generator(func, data, model_data['topology'][0], model_data['topology'][-1][1])
    

    model = network_builder(model_data['topology'], 
                                     model_data['op_sets'], 
                                     None, 
                                     None, 
                                     None, 
                                     None, 
                                     model_data['output_activation'], 
                                     model_data['use_bias'])
    
    model.compile('sgd','mse', metrics)
    model = misc.set_model_weights(model, model_data['weights'])
        
    gen, steps = func(data)
    result = model.evaluate_generator(gen, steps)
    performance = {}
    
    for idx, metric in enumerate(model.metrics_names):
        performance[metric] = result[idx]
        
    if special_metrics:
        p = misc.evaluate_special_metrics(model, special_metrics, gen, steps)
        for idx, metric in enumerate(special_metrics):
            performance[metric.__name__] = p[idx]
            
    from keras import backend as K
    K.clear_session()
    del K
        
    return performance

def predict(model_data, func, data):     
    """Construct the network and generate prediction
    
    Args:
        model_data (dict): Data of the network
        func (callable): Function to produce the generator and #mini-batch
        data: Input to func
        
    Returns:
        Numpy array of predicted output
        
    """
        
    if model_data is None:
        raise Exception('No model exist for evaluation')
    
    misc.test_generator(func, data, model_data['topology'][0])
    

    model = network_builder(model_data['topology'], 
                                     model_data['op_sets'], 
                                     None, 
                                     None, 
                                     None, 
                                     None, 
                                     model_data['output_activation'], 
                                     model_data['use_bias'])
    
    model.compile('sgd','mse', ['mse',])
    model = misc.set_model_weights(model, model_data['weights'])
    
    gen, steps = func(data)
    return model.predict_generator(gen, steps)


def load(filename, model_data_attributes, model_name):
    """Load a pretrained model from disk with the given filename
    
    Args:
        filename (str): Path to model data
        model_data_attributes (list): Attributes required from the file
        model_name (str): Name of the model to check compatibility
        
    Returns:
        model_data (dict): Data of the pretrained model
        
    """
    
    try:
        fid = open(filename,'rb')
        model_data = dill.load(fid)
        fid.close()
    except:
        raise Exception('Cannot open model data from ´´filename´´' % filename)
    
    
    model_data_attributes = model_data.keys()
    for key in model_data_attributes:
        if not key in model_data_attributes:
            raise Exception('given model data lacks attribute ´´%s´´' % key)
    
    
    if model_data['model'] != model_name:
        raise Exception('given model data is an instance of ´´%s´´, not %s model' % (model_data['model'], model_name))
    
    model = None    

    try:
        model = network_builder(model_data['topology'], 
                                          model_data['op_sets'], 
                                          None, 
                                          None, 
                                          None, 
                                          None, 
                                          model_data['output_activation'], 
                                          model_data['use_bias'])
        
        model.compile('sgd','mse',['mse',])
    except:
        raise Exception('Cannot build model from model data')
    
    try:
        model = misc.set_model_weights(model, model_data['weights'])
    except:
        raise Exception('Given model weights is incompatible with given topology')
        
    if model is not None:
        from keras import backend as K
        K.clear_session()
        del K
    
    return model_data
