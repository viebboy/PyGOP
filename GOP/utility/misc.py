#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Miscellaneous functions shared by all GOP-based models


Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

from __future__ import print_function

import os, sys, dill, inspect, copy, numpy as np
dill.settings['recurse'] = True
try:
    import cPickle as pickle
except ImportError:
    import pickle
    
from keras import backend as K
from keras.callbacks import Callback as Callback
from . import gop_operators

class BestModelLogging(Callback):
    """Custom callback used when training Keras model
    
    This callback allows recording all metrics in keras model
    as well as special metrics, which require full batch evaluation instead of
    mini-batch aggregation
    
    Args:
        convergence_measure (str): Name of metric to monitor best model setting
        direction (str): String to indicate relative quality of convergence measure
        special_metrics (list): List of callable to calculate special metrics
        train_func (function): Function to return train generator and #mini-batches
        train_data: Input to train_func
        val_func (function): Function to return val generator and #mini-batches
        val_data: Input to val_func
        test_func (function): Function to return test generator and #mini-batches
        test_data: Input to test_func
        
    Attributes:
        measure (float): Best performance according to the convergence_measure
        performance (dict): History of all metrics for all epochs
        model_weights (list): Weights of the model to achieve best performance (measure)
        
    """
 
    def __init__(self, 
                 convergence_measure,
                 direction,
                 special_metrics=None,
                 train_func=None,
                 train_data=None,
                 val_func=None,
                 val_data=None,
                 test_func=None,
                 test_data=None):
        

        self.convergence_measure = convergence_measure    
        self.direction = direction
        self.special_metrics = special_metrics
        self.performance = {}
        self.train_gen, self.train_steps = train_func(train_data) if train_func else (None, None)
        self.val_gen, self.val_steps = val_func(val_data) if val_func else (None, None)
        self.test_gen, self.test_steps = test_func(test_data) if test_func else (None, None)
        self.special_metrics_names = [metric.__name__ for metric in special_metrics] if special_metrics else []
        
        
    def on_train_begin(self, logs={}):
        self.model_weights = None
        if self.direction == 'higher':
            self.measure = np.NINF
            self.sign = 1.0
        else:
            self.measure = np.inf
            self.sign = -1.0 
        
        for metric in self.model.metrics_names + self.special_metrics_names:
            self.performance['train_' + metric] = []
            self.performance['val_' + metric] = [] if self.val_gen else None
            self.performance['test_' + metric] = [] if self.test_gen else None
                
        self.output_shape = self.model.layers[-1].output_shape[1:]
        
        return
  
    def on_epoch_end(self, epoch, logs={}):
        # log standard metrics
        for metric in self.model.metrics_names:
            self.performance['train_' + metric].append(logs[metric])
        
            if self.val_gen:
                self.performance['val_' + metric].append(logs['val_' + metric])
            
        if self.test_gen:
            test_performance = self.model.evaluate_generator(self.test_gen, self.test_steps)
            for idx, metric in enumerate(self.model.metrics_names):
                self.performance['test_' + metric].append(test_performance[idx])
        
        
        # log special metrics
        if self.special_metrics:
            train_performance = evaluate_special_metrics(self.model, self.special_metrics, self.train_gen, self.train_steps)
            val_performance = evaluate_special_metrics(self.model, self.special_metrics, self.val_gen, self.val_steps) if self.val_gen else None
            test_performance = evaluate_special_metrics(self.model, self.special_metrics, self.test_gen, self.test_steps) if self.test_gen else None
            
            for idx, metric in enumerate(self.special_metrics_names):
                self.performance['train_' + metric].append(train_performance[idx])
                if self.val_gen:
                    self.performance['val_' + metric].append(val_performance[idx])
                if self.test_gen:
                    self.performance['test_' + metric].append(test_performance[idx])
                 
        if self.sign*(self.performance[self.convergence_measure][-1] - self.measure) > 0:
            self.measure = self.performance[self.convergence_measure][-1]
            self.model_weights = self.model.get_weights()
        
        return


def get_he_init(fan_in, fan_out):
    """He initializer method
    
    Args:
        fan_in (int): input dimension
        fan_out (int): output dimension
        
    Returns:
        Callable that allows the initialization with given shape
        
    """
    scale=1.0/max(1.0,float(fan_in+fan_out)/2)
    stddev=np.sqrt(scale)

    def f(shape,dtype=None):
        return K.truncated_normal(shape,stddev=stddev,dtype=dtype)
    return f


def initialize_states(parameters, model):
    """Initialize train states of the given model using the given parameters
    
    This function will check if train_states exists in the temporary dir, load
    the previous state if conditions match, create fresh train_states otherwise
    
    Args:
        parameters (dict): Model parameters
        model (str): Name of the model
        
    Returns:
        train_states (dict): Current model configuration
    
    """

    def get_train_states():
        return {"layer_iter":   0,
                "block_iter":   0,
                "op_set_indices":       {},
                "weights":      {},
                'use_bias':     parameters['use_bias'],
                'output_activation': parameters['output_activation'] if 'output_activation' in parameters.keys() else None,
                "topology":     [parameters['input_dim'],],
                "history":      [],
                'measure':      [],
                "is_finished":  False,
                'model':   model}
    
    path = os.path.join(parameters["tmp_dir"], parameters["model_name"])
    
    if os.path.isdir(path):
        file_path = os.path.join(path, 'train_states.pickle')
        
        if os.path.exists(file_path):
            fid = open(file_path, 'r')
            train_states = pickle.load(fid)
            fid.close()
            
            if train_states['model'] != model or \
            train_states['topology'][0] != parameters['input_dim'] or \
            len(train_states['topology']) == 1 or \
            train_states['topology'][-1][1] != parameters['output_dim'] or \
            train_states['use_bias'] != parameters['use_bias'] or \
            train_states['output_activation'] != parameters['output_activation']:
                train_states = get_train_states()
        else:
            train_states = get_train_states()
    
    else:
        raise Exception('directory "%s" does not exist!' % path)
    
    return train_states

def get_batchjob_parameters_keys():
    compulsory = ['name','mem', 'core', 'partition', 'time', 'no_machine', 'python_cmd']
    noncompulsory = ['configuration', 'constraint']
    
    return compulsory, noncompulsory

def setup_batchjob_file(params, cmd, path):
    """Setup content of batch job file on SLURM cluster using the given settings
    
    Args:
        params (dict): Batch job file parameters
        cmd (str): Command to run python on the cluster
        path (str): Temporary directory that holds output, error files
        
    Returns:
        contents (list): Each element is one row in the batch job file
        
    """
    
    contents = []
    contents.append('#!/bin/bash \n')
    contents.append('#SBATCH -J %s \n' % params['name'])
    contents.append('#SBATCH -o %s' % os.path.join(path, params['name']) + '_%a.o\n')
    contents.append('#SBATCH -e %s' % os.path.join(path,params['name']) + '_%a.e\n')
    contents.append('#SBATCH --mem=%dG \n' % params['mem'])
    contents.append('#SBATCH -c %d \n' % params['core'])
    contents.append('#SBATCH -p %s \n' % params['partition'])
    contents.append('#SBATCH -t %s \n' % params['time'])
    contents.append('#SBATCH --array=0-%s \n' % str(params['no_machine']-1))
    if 'constraint' in params.keys():
        contents.append('#SBATCH --constraint=%s \n' % params['constraint'])
    
    if 'configuration' in params.keys():
        contents.append(params['configuration'] + '\n')
        
    contents.append('\n')
    contents.append(cmd)
    
    return contents

def get_batchjob_parameters_description():
    """Produce description of the batch job parameters required
    
    """
    bj_param_description =  '"name" (str): name of the job submitted to SLURM \n' +\
                                    '"mem" (int): the memory size of each node (in GB) \n' +\
                                    '"core" (int): number of cores of each node\n' +\
                                    '"partition" (str): name of the partition in the cluster \n' +\
                                    '"time" (str): the maximum time to run a node, e.g., "7-00:00:00" is 7 days \n' +\
                                    '"no_machine" (int): number of machines (node) to use \n' +\
                                    '"configuration" (str): the configuration settings, e.g. module loading etc written in string \n' +\
                                    '"python_cmd" (str): command to run python in string, e.g., "srun python" \n' +\
                                    '"constraint" (str): constraint parameter of the cluster, e.g. architecture of cpu or gpu \n'
                                    
    return bj_param_description

def pickle_generator(filename, train_func, train_data, val_func, val_data, test_func, test_data):
    """Pickle the functions that produce generator and steps and dump to disk 
    
    Args:
        filename (str): Filename on disk to dump the function data
        train_func (function): Function to return train generator and #mini-batches
        train_data: Input to train_func
        val_func (function): Function to return val generator and #mini-batches
        val_data: Input to val_func
        test_func (function): Function to return test generator and #mini-batches
        test_data: Input to test_func
        
    Return:
        
    """
    if not os.path.exists(filename):
        data = {'train_func': dill.dumps(train_func, recurse=True),
        'train_data' : train_data,
        'val_func': dill.dumps(val_func, recurse=True) if val_func else None,
        'val_data' : val_data,
        'test_func': dill.dumps(test_func, recurse=True) if test_func else None,
        'test_data' : test_data,
        'train_func_path' : os.path.dirname(inspect.getabsfile(train_func)),
        'val_func_path': os.path.dirname(inspect.getabsfile(val_func)) if val_func else None,
        'test_func_path' : os.path.dirname(inspect.getabsfile(test_func)) if test_func else None}
            
        with open(filename,'wb') as fid:
            pickle.dump(data, fid)
            
    return 

def unpickle_generator(filename):
    """Load the functions to produce data generators from disk
    
    Args:
        filename (str): Path to the pickled functions
        
    Returns:
        train_func (function): Function to return train generator and #mini-batches
        train_data: Input to train_func
        val_func (function): Function to return val generator and #mini-batches
        val_data: Input to val_func
        test_func (function): Function to return test generator and #mini-batches
        test_data: Input to test_func
        
    """

    with open(filename,'rb') as f:
        data = pickle.load(f)
        
    train_gen_path = data['train_func_path']
    val_gen_path = data['val_func_path']
    test_gen_path = data['test_func_path']
    
    if not train_gen_path in sys.path: 
        sys.path.append(train_gen_path)
    
    if val_gen_path and not val_gen_path in sys.path: 
        sys.path.append(val_gen_path)
    
    if test_gen_path and not test_gen_path in sys.path: 
        sys.path.append(test_gen_path)
    
    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
        
    train_func = dill.loads(data['train_func']) if data['train_func'] else None
    val_func = dill.loads(data['val_func']) if data['val_func'] else None
    test_func = dill.loads(data['test_func']) if data['test_func'] else None


    return train_func, train_data, val_func, val_data, test_func, test_data

def get_op_set_index(no_op_set, no_machine, machine_no):
    """Calculate the start and end operator set index that are given to a machine
    
    Args:
        no_op_set: Number of total operator sets
        no_machine: Number of machines used
        machine_no: Machine index
        
    Returns:
        start_idx (int): Start index in the list of operator sets
        stop_idx (int): Stop index in the list of operator sets 
    
    """
    
    assert no_machine <= no_op_set, 'Number of machines (%d) larger than number of operator sets (%d)' %(no_machine, no_op_set)
    op_set_per_machine = int(np.ceil(no_op_set / float(no_machine)))
    start_idx = machine_no*op_set_per_machine
    stop_idx = min(no_op_set, (machine_no+1)*op_set_per_machine)
    
    return start_idx, stop_idx
    
    
def evaluate(model, train_gen, train_steps, val_gen, val_steps, test_gen, test_steps):
    """Evaluate the model using given data generators
    
    Args:
        model: Keras model instance
        train_gen (generator): Train generator
        train_steps (int): Number of mini-batches
        val_gen (generator): Validation generator
        val_steps (int): Number of mini-batches
        test_gen (generator): Test generator
        test_steps (int): Number of mini-batches
        
    Returns:
        performance (dict): Performances according to the metrics inside the Keras model
        
    """
    
    train_performance = model.evaluate_generator(train_gen, train_steps)
    val_performance = model.evaluate_generator(val_gen, val_steps) if val_gen else None
    test_performance = model.evaluate_generator(test_gen, test_steps) if test_gen else None
    
    performance = {}
    for index, metric in enumerate(model.metrics_names):
        performance['train_' + metric] = train_performance[index]
        performance['val_' + metric] = val_performance[index] if val_performance else None
        performance['test_' + metric] = test_performance[index] if test_performance else None
    
    return performance

def evaluate_special_metrics(model, special_metrics, gen, steps):
    """Evaluate the model using given data generators with special metrics
    
    Args:
        model: Keras model instance
        special_metrics (list): List of callable of special metrics
        gen (generator): Train generator
        steps (int): Number of mini-batches
        
    Returns:
        performance (dict): Performances according to the given special metrics
        
    """
    
    y_true = np.zeros([1,] + list(model.layers[-1].output_shape[1:]), dtype=np.float32)
    y_pred = np.zeros([1,] + list(model.layers[-1].output_shape[1:]), dtype=np.float32)
    performance = []
    
    for _ in range(steps):
        x, y = next(gen)
        y_true = np.concatenate((y_true, y), axis=0)
        y_pred = np.concatenate((y_pred, model.predict(x)), axis=0)
    
    y_true, y_pred = y_true[1:], y_pred[1:]
    
    for metric in special_metrics:
        performance.append(metric(y_true, y_pred))
        
    return performance


def check_convergence(new_measure, old_measure, direction, threshold):
    """Check if the performance meets the given threshold
    
    Args:
        new_measure (float): New performance
        old_measure (float): Old performance
        direction (str): String to indicate how to compare two measures
        threshold (float): The given threshold
        
    Returns:
        True if the new measure satisfies threshold, False otherwise
        
    """
    
    sign = 1.0 if direction == 'higher' else -1.0
    
    if sign*(new_measure - old_measure) / old_measure < threshold:
        return True
    else:
        return False
    
def set_model_weights(model, weights):
    """Set the given weights to keras model
    
    Args:
        model : Keras model instance
        weights (dict): Dictionary of weights
    
    Return:
        Keras model instance with weights set
        
    """
    for key in weights.keys():    
        model.get_layer(key).set_weights(weights[key])
        
    return model

    
def test_generator(func, data, input_dim=None, output_dim=None):
    """Test if the given function to produce data generator works
    
    Produce the generators and try to produce data for 10 mini batches
    Raise exception if dimensions mismatch or generator not working
    
    Args:
        func (callable): Function to produce generator
        data: Input to func
        input_dim (int): expected dimension of input
        output_dim (int): optional expected dimension of output
        
    Return:
        
        
    """
    if isinstance(input_dim,int):
        input_dim = (input_dim,)
    if isinstance(output_dim, int):
        output_dim = (output_dim,)
        
    gen, steps = func(data)
    steps = min(steps, 10)
    for _ in range(steps):
        x = next(gen)
        if len(x) == 0:
            raise Exception('output produced by data generator is empty!')
        elif len(x) == 1:
            assert x[0].shape[1:] == input_dim, \
            'input dimension of generated data does not match input dimension of model'
        elif len(x) == 2:
            assert x[0].shape[1:] == input_dim and x[1].shape[1:] == output_dim, \
            'output dimension of generated data does not match output dimension of model'

        
    return

def pickle_custom_metrics(metrics, filename):
    """Pickle the metrics if there is callable in the list of metrics
    
    Args:
        metrics (list): List of metrics
        filename (str): Path to dump the pickled file
        
    Return:
        
    """

    metric_callable = False
    for metric in metrics:
        if callable(metric):
            metric_callable = True
            break
        
    if metric_callable and not os.path.exists(filename):
        metric_names = []
        function_string = []
        function_path = []
        count = 0
        for metric in metrics:
            if isinstance(metric, str):
                metric_names.append(metric)
            elif callable(metric):
                metric_names.append(count)
                function_string.append(dill.dumps(metric, recurse=True))
                
                if not os.path.dirname(inspect.getabsfile(metric)) in function_path:
                    function_path.append(os.path.dirname(inspect.getabsfile(metric)))
                    
                count += 1
                
        metric_data = {'names':metric_names,
                       'function_string': function_string,
                       'function_path': function_path}
        
        with open(filename, 'wb') as fid:
            pickle.dump(metric_data, fid)

    return metric_callable

def unpickle_custom_metrics(filename):
    """Load the pickled metrics 
    
    Args:
        filename (str): Path to the pickled file
        
    Returns:
        metrics (list): List of metrics
        
    """
    metrics = []
    with open(filename, 'rb') as fid:
        metric_data = pickle.load(fid)
        
    names = metric_data['names']
    function_string = metric_data['function_string']
    function_path = metric_data['function_path']
    
    for path in function_path:
        if not path in sys.path:
            sys.path.append(path)
    
    for metric in names:
        if isinstance(metric, str):
            metrics.append(metric)
        elif isinstance(metric, int):
            metrics.append(dill.loads(function_string[metric]))
    
    return metrics

def pickle_custom_loss(loss, filename):
    """Pickle the loss if it is a callable
    
    Args:
        loss : loss function or a string of the function name
        filename (str): Path to dump the pickled file
        
    Return:
        True if loss is callable, False otherwise
        
    """
    
    if callable(loss) and not os.path.exists(filename):
        custom_loss = {}
        custom_loss['function_string'] = dill.dumps(loss, recurse=True)
        custom_loss['path'] = os.path.dirname(inspect.getabsfile(loss))
        
        with open(filename, 'wb') as fid:
            pickle.dump(custom_loss, fid)
        
    if callable(loss):
        return True
    else:
        return False


def unpickle_custom_loss(filename):
    """Load the pickled loss function
    
    Args:
        filename (str): Path to the pickled file
        
    Returns:
        loss (callable): the loss function
        
    """
    
    with open(filename, 'rb') as fid:
        custom_loss = pickle.load(fid)
        
    if not custom_loss['path'] in sys.path:
        sys.path.append(custom_loss['path'])
    
    return dill.loads(custom_loss['function_string'])
    

def pickle_special_metrics(metrics, filename):
    """Pickle special metrics and dump to filename
    
    Args:
        metrics (list): List of special metrics
        filename (str): Path to the pickled file
        
    Return:
        True if special metric is not None, False otherwise
        
    """
    
    if metrics is not None and not os.path.exists(filename):
        data = {'function_string':[],
                'path':[]}
        
        for metric in metrics:
            data['function_string'].append(dill.dumps(metric, recurse=True))
            if not os.path.dirname(inspect.getabsfile(metric)) in data['path']:
                data['path'].append(os.path.dirname(inspect.getabsfile(metric)))
        
        with open(filename, 'wb') as fid:
            pickle.dump(data, fid)
        
    if metrics is not None:
        return True
    else:
        return False

def unpickle_special_metrics(filename):
    """Load the pickled special metrics
    
    Args:
        filename (str): Path to the pickled file
        
    Returns:
        special_metrics (list): List of callable
        
    """
    
    metrics = []
    with open(filename, 'rb') as fid:
        data = pickle.load(fid)
        for path in data['path']:
            if not path in sys.path:
                sys.path.append(path)
                
        for function_str in data['function_string']:
            metrics.append(dill.loads(function_str))
    
    return metrics

def pickle_custom_operators(nodal_set, pool_set, activation_set, filename):
    """Pickle and dump custom operators if exists
    
    Args:
        nodal_set (list): List of nodal operators
        pool_set (list): List of pooling operators
        activation_set (list): List of activation operators
        filename (str): Path to the pickled file on disk
        
    Returns:
        True if exists custom operators else False
        
    """
    
    if os.path.exists(filename):
        return True
    
    has_custom_operators = False
    for operator in nodal_set + pool_set + activation_set:
        if callable(operator):
            has_custom_operators = True
            break
    
    if has_custom_operators:
        count = 0
        pickled_func = []
        nodal_set_ = []
        pool_set_ = []
        activation_set_ = []
        
        for nodal in nodal_set:
            if callable(nodal):
                nodal_set_.append(count)
                count += 1
                pickled_func.append(dill.dumps(nodal, recurse=True))
            else:
                nodal_set_.append(nodal)
                
        for pool in pool_set:
            if callable(pool):
                pool_set_.append(count)
                count += 1
                pickled_func.append(dill.dumps(pool, recurse=True))
            else:
                pool_set_.append(pool)
                
        for activation in activation_set:
            if callable(activation):
                activation_set_.append(count)
                count += 1
                pickled_func.append(dill.dumps(activation, recurse=True))
            else:
                activation_set_.append(activation)
        
        operators = {'nodal_set': nodal_set_,
                     'pool_set': pool_set_,
                     'activation_set': activation_set_,
                     'pickled_func': pickled_func}
        
        with open(filename, 'wb') as fid:
            pickle.dump(operators, fid)
        
    return has_custom_operators


def unpickle_custom_operators(filename):
    """Load the custom operators from pickled file
    
    """
    
    nodal_set = []
    pool_set = []
    activation_set = []
    
    with open(filename, 'rb') as fid:
        operators = pickle.load(fid)
    
    pickled_func = operators['pickled_func']
    
    for nodal in operators['nodal_set']:
        if isinstance(nodal, str):
            nodal_set.append(nodal)
        else:
            nodal_set.append(dill.loads(pickled_func[nodal]))
    
    for pool in operators['pool_set']:
        if isinstance(pool, str):
            pool_set.append(pool)
        else:
            pool_set.append(dill.loads(pickled_func[pool]))
            
    for activation in operators['activation_set']:
        if isinstance(activation, str):
            activation_set.append(activation)
        else:
            activation_set.append(dill.loads(pickled_func[activation]))
            
    return nodal_set, pool_set, activation_set

def get_gpu_str(devices):
    """Produce string of GPUs given the list of int
    
    """
    gpu_str = ''
    for gpu in devices:
        gpu_str += str(gpu) + ','
    return gpu_str[:-1]

def partition_indices(start_idx, stop_idx, no_partition):
    """Partition the range of indices according to given number of partitions
    
    Args:
        start_idx (int): Start index in the range
        stop_idx (int): Stop index in the range
        no_partition (int): Number of partitions
        
    Return:
        start_indices (list): list of start indices
        stop_indices (list): List of stop indices
        
    """
    assert stop_idx > start_idx, 'stop_idx (%d) must be larger than start_idx (%d)' %(stop_idx, start_idx)
    
    start_indices = []
    stop_indices = []
    
    if no_partition > stop_idx - start_idx:
        no_partition = stop_idx - start_idx
    
    step = int(np.ceil(stop_idx-start_idx)/float(no_partition))
    
    for k in range(no_partition):
        start_indices.append(start_idx + k*step)
        stop_indices.append(min(stop_idx, start_idx + (k+1)*step))
        
    return start_indices, stop_indices

def remove_files(filenames):
    """Function to remove the files given in filenames
    
    """
    for f in filenames:
        try:
            os.remove(f)
        except:
            raise Exception('Cannot remove temporary file "%s"' % f)
            
def dump_data(params,
              train_states,
              train_func,
              train_data,
              val_func,
              val_data,
              test_func,
              test_data):
    """Pickle and dump the model parameters, current train states and data functions
    
    Args:
        params (dict): Model parameters
        train_states (dict): Current topology configuration
        train_func (function): Function to return train generator and #mini-batches
        train_data: Input to train_func
        val_func (function): Function to return val generator and #mini-batches
        val_data: Input to val_func
        test_func (function): Function to return test generator and #mini-batches
        test_data: Input to test_func
        
    Returns:
        
        
    """
    
    
    path = os.path.join(params['tmp_dir'], params['model_name'])
    
    params_filename = os.path.join(path, 'params.pickle')
    train_states_filename = os.path.join(path, 'train_states_tmp.pickle')
    data_filename = os.path.join(path, 'data.pickle')
    custom_loss_filename = os.path.join(path, 'custom_loss.pickle')
    custom_metrics_filename = os.path.join(path, 'custom_metrics.pickle')
    special_metrics_filename = os.path.join(path, 'special_metrics.pickle')
    custom_operator_filename = os.path.join(path, 'custom_operators.pickle')
    
    # dump train_states
    with open(train_states_filename, 'wb') as fid:
        pickle.dump(train_states, fid)
            
    # dump custom loss, metrics
    pickle_generator(data_filename, train_func, train_data, val_func, val_data, test_func, test_data)
    has_custom_loss = pickle_custom_loss(params['loss'], custom_loss_filename)
    has_custom_metrics = pickle_custom_metrics(params['metrics'], custom_metrics_filename)
    has_special_metrics = pickle_special_metrics(params['special_metrics'], special_metrics_filename)
    has_custom_operators = pickle_custom_operators(params['nodal_set'], params['pool_set'], params['activation_set'], custom_operator_filename)
    
    if not os.path.exists(params_filename):
        params_ = copy.deepcopy(params)
        if has_custom_loss:
            params_['loss'] = []
        if has_custom_metrics:
            params_['metrics'] = []
        if has_special_metrics:
            params_['special_metrics'] = []
        if has_custom_operators:
            params_['nodal_set'] = []
            params_['pool_set'] = []
            params_['activation_set'] = []
            
        with open(params_filename,'wb') as fid:
            pickle.dump(params_, fid)
    
    return 

def map_operator_from_index(op_set_indices, nodal_set, pool_set, activation_set):
    """Map operator set indices to actual operator set
    
    Args:
        op_set_indices (dict): dictionary of operator set indices of all blocks
        nodal_set (list): List of all nodal operators
        pool_set (list): List of all pooling operators
        activation_set (list): List of all activation operators
        
    Returns:
        op_sets (dict): dictionary of operator sets with same keys
        
    """
    all_op_sets = gop_operators.get_op_set(nodal_set, pool_set, activation_set)
    
    op_sets = {}
    
    for key in op_set_indices.keys():
        op_sets[key] = all_op_sets[op_set_indices[key]]
        
    return op_sets

def check_model_parameters(params, default_params):
    """Perform sanity check on model parameters 
    
    Args:
        params (dict): The given paramters
        default_params (dict): Default parameters of the model
        
    Returns:
        verified params with added default values if missing
        
    """
    keys = params.keys()
    assert os.path.exists(params['tmp_dir']), 'Given temporary directory "%s" does not exist!' % params['tmp_dir']
    assert 'tmp_dir' in keys, 'This model requires a read/writeable temporary directory during computation, please specify in "tmp_dir"'
    assert 'model_name' in keys, 'This model requires a unique name in temporary directory (tmp_dir), please specify in "model_name"'
    assert os.access(params['tmp_dir'], os.W_OK), 'The given temporarary directory "%s" does not have write permission ' % params['tmp_dir']
    
    
    if not os.path.exists(os.path.join(params['tmp_dir'], params['model_name'])):
        os.mkdir(os.path.join(params['tmp_dir'], params['model_name']))
        
    # verify input_dim, output_dim
    assert 'input_dim' in keys, 'Input dimension should be specified through "input_dim"'
    assert 'output_dim' in keys, 'Output dimension should be specified through "output_dim"'
    
    # setup default parameters if necessary
    
    for key in default_params.keys():
        if not key in keys:
            params[key] = default_params[key]
            
    # verify operator set
    gop_operators.has_valid_operator(params['nodal_set'], params['pool_set'], params['activation_set'])
    params['no_op_set'] = len(params['nodal_set']) * len(params['pool_set']) * len(params['activation_set'])
    
    # verify cluster option
    if params['cluster']:
        description = get_batchjob_parameters_description()
        assert 'batchjob_parameters' in keys, 'When using SLURM cluster, a dictionary of batchjob parameters must be given via the key "batchjob_parameters" \n' +\
                                              '------------------------------------------------------------------------------------------------------------ \n' +\
                                                        'batchjob_parameters should have the following (key, value): \n%s' % description
                                                    
    
        for key in get_batchjob_parameters_keys()[0]:
            if not key in params['batchjob_parameters'].keys():
                raise 'please specify parameter "%s" in "batchjob_parameters"' % key
        
        no_op_set = len(params['nodal_set']) * len(params['pool_set']) * len(params['activation_set'])
        if params['batchjob_parameters']['no_machine'] > no_op_set:
            print('Number of machines used in cluster is larger than number of operator sets, ' + \
                  'setting "no_machine" equal to number of operator sets now')
            params['batchjob_parameters']['no_machine'] = no_op_set
        
    # verify computation devices
    assert isinstance(params['search_computation'], (list, tuple)), '"search_computation" should be given as a tuple or list ' +\
            'with 1st element being "cpu" or "gpu" and 2nd element being number of parallel processes when using "cpu" or a list of GPU numbers when using "gpu"'
            
    if params['search_computation'][0] == 'cpu':
        assert isinstance(params['search_computation'][1], int), 'When "search_computation"="cpu", specify number of parallel processes to used as an integer'
    elif params['search_computation'][0] == 'gpu':
        assert isinstance(params['search_computation'][1], (list, tuple)), 'When "search_computation"="gpu", specify list of gpu number (int) to use'
        
    if params['finetune_computation'][0] == 'gpu':
        assert isinstance(params['finetune_computation'][1], (list, tuple)), 'When "finetune_computation"="gpu", specify list of gpu number (int) to use'
        
    if not 'output_activation' in keys:
        params['output_activation'] = None
    
    assert isinstance(params['metrics'], (str, list, tuple)), 'metrics should be given as a single/list/tuple of (callable or string)'
    if isinstance(params['metrics'], str):
        params['metrics'] = [params['metrics'],]
        
    if params['special_metrics'] is not None:
        assert isinstance(params['special_metrics'], (list, tuple)), 'special_metrics should be given as a list/tuple of callable'
    
        for metric in params['special_metrics']:
            assert callable(metric), 'special_metrics should be a list/tuple of callable'
        
    # correct the metric names if abbreviation is used
    if 'mse' in params['metrics']:
        params['metrics'][params['metrics'].index('mse')] = 'mean_squared_error'
        
    if params['convergence_measure'] == 'mse':
        params['convergence_measure'] = 'mean_squared_error'
        
    if 'accuracy' in params['metrics']:
        params['metrics'][params['metrics'].index('accuracy')] = 'acc'
    
    if params['convergence_measure'] == 'accuracy':
        params['convergence_measure'] = 'acc'

    
    if 'mae' in params['metrics']:
        params['metrics'][params['metrics'].index('mae')] = 'mean_absolute_error'
    
    if params['convergence_measure'] == 'mae':
        params['convergence_measure'] = 'mean_absolute_error'
    
    
    assert isinstance(params['convergence_measure'], str), 'convergence_measure should be a string,' +\
        'in case it is a custom metric or special metric given as a function, convergence_measure should be the function name'
        
    # verify if convergence_measure in metrics
    metrics = []
    for m in params['metrics']:
        if callable(m): 
            metrics.append(m.__name__)
        else:
            metrics.append(m)
    
    if params['special_metrics'] is not None:
        for m in params['special_metrics']:
            metrics.append(m.__name__)
    
    assert params['convergence_measure'] in metrics, 'convergence_measure "%s" should belong to the list of metrics or special_metrics' % params['convergence_measure']
    
    assert isinstance(params['loss'], str) or callable(params['loss']), 'loss should be a string indicate the loss function name or a tensorflow/keras function'
    
    assert params['direction'] in ['higher', 'lower'], 'direction should be ("higher"/"lower") indicating whether higher/lower value of convergence measure is better'
    
    return params
