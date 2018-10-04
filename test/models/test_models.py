#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test different model with different parameter settings and different computation environments


Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

from __future__ import print_function
import os, sys, getopt, shutil, numpy as np, tensorflow as tf
import test_utility as utils


INPUT_DIM = 30
OUTPUT_DIM = 10

def data_loading_func(filenames):
    x = np.load(filenames[0])
    y = np.load(filenames[1])
    steps = 10
    def gen():
        while True:
            for i in range(steps):
                start_idx = i*100
                stop_idx = min(1000, (i+1)*100)
                yield utils.scale(x[start_idx:stop_idx]).astype('float32'), y[start_idx:stop_idx].astype('float32')
                
    return gen(), steps


def predict_func(filename):
    x = np.load(filename)
    steps = 10
    def gen():
        while True:
            for i in range(steps):
                start_idx = i*100
                stop_idx = min(1000, (i+1)*100)
                yield utils.scale(x[start_idx:stop_idx])
                
    return gen(), steps


def get_computation_setup(classifier, option):
    
    no_machine = 1
    if option == 0:
        model_name = 'test_' + classifier + '_local_cpu'
        search_computation = ('cpu',8) 
        finetune_computation = ('cpu',8)
        cluster = False
    elif option == 1:
        model_name = 'test_' + classifier + '_cluster_cpu'
        cluster = True
        search_computation = ('cpu',4) 
        finetune_computation = ('cpu',4)
        no_machine = 4
    elif option == 2:
        model_name = 'test_' + classifier + '_cluster_gpu'
        cluster = True
        search_computation = ('gpu',[0,1])
        finetune_computation = ('cpu',4)
        no_machine = 2
    elif option == 3:
        cluster = False
        model_name = 'test_' + classifier + '_local_gpu'
        search_computation = ('gpu',[0,1]) 
        finetune_computation = ('gpu', [0,1])
        
    return no_machine, model_name, search_computation, finetune_computation, cluster

def prepare_test_directory(model_name, version):
    cwd = os.getcwd()
    test_dir = os.path.join(cwd, model_name + '_' + version)
    
        
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    else:
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
    
    data_dir = os.path.join(test_dir, 'data')
    os.mkdir(data_dir)
    
    X = [np.random.rand(1000, INPUT_DIM) for i in range(3)]
    Y = [np.random.rand(1000, OUTPUT_DIM) for i in range(3)]
    suffix = ['_train', '_val', '_test']
    for i in range(3):
        np.save(os.path.join(data_dir, 'x' + suffix[i] + '.npy'), X[i])
        np.save(os.path.join(data_dir, 'y' + suffix[i] + '.npy'), Y[i])
        
    return test_dir, data_dir

def load_data(data_dir):
    
    x_train = os.path.join(data_dir, 'x_train.npy')
    y_train = os.path.join(data_dir, 'y_train.npy')
    x_val = os.path.join(data_dir, 'x_val.npy')
    y_val = os.path.join(data_dir, 'y_val.npy')
    x_test = os.path.join(data_dir, 'x_test.npy')
    y_test = os.path.join(data_dir, 'y_test.npy')
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def setup_parameters(params, classifier, option, version, cluster_name):
    
    if classifier == 'hemlgop':
        params['max_layer'] = 4
        params['max_block'] = 4
    elif classifier == 'homlgop':
        params['max_layer'] = 4
        params['max_block'] = 4
    elif classifier == 'hemlrn':
        params['max_layer'] = 4
        params['max_block'] = 4
    elif classifier == 'homlrn':
        params['max_layer'] = 4
        params['max_block'] = 4
    elif classifier in ['popfast', 'popmemo', 'popmemh', 'pop']:
        params['max_topology'] = [40, 40, 40, 40]
    else:
        raise Exception('Model not recognised')
        
    no_machine, model_name, search_computation, finetune_computation, is_cluster = get_computation_setup(classifier, 
                                                                                                      option) 
    
    test_dir, data_dir = prepare_test_directory(model_name, version)
    

    params['input_dim'] = INPUT_DIM
    params['output_dim'] = OUTPUT_DIM
    params['tmp_dir'] = test_dir
    params['model_name'] = model_name
    params['special_metrics'] = [utils.special_metric,]
    params['metrics'] = ['mae','mse', utils.tf_mae]
    params['cluster'] = is_cluster
    params['search_computation'] = search_computation
    params['finetune_computation'] = finetune_computation
    
    if is_cluster:
        if cluster_name == 'narvi':
            if option == 2:
                raise Exception('Not yet support narvi gpu')
                
            partition = 'normal'
            if version=='py2':
                configuration = 'source activate py2_test_cpu'
            else:
                configuration = 'source activate py3_test_cpu'
            
            python_cmd = 'python'
        
        elif cluster_name == 'taito':
            python_cmd = 'srun python'
            if option in [0,1]:
                partition = 'parallel'
            else:
                partition = 'gpu'
            
            if option in [0,1]:   
                if version=='py2':
                    configuration = 'module purge \nmodule load python-env/2.7.10 \n'
                else:
                    configuration = 'module purge \nmodule load python-env/3.5.3 \n'
            else:
                configuration = '#SBATCH --gres=gpu:p100:2 \n'
                if version == 'py2':
                    configuration += 'module purge \nmodule load python-env/2.7.10-ml \n'
                else:
                    configuration += 'module purge \nmodule load python-env/3.5.3-ml \n'
            
        else:
            partition = ''
            python_cmd = ''
            configuration = ''
            raise Exception('Only support taito or narvi cluster')
                
        batchjob_parameters = {'name': 'bj',
                               'mem':16,
                               'core':4,
                               'partition':partition,
                               'time':'1:00:00',
                               'no_machine':no_machine,
                               'constraint':'hsw' if option in [0,1] else 'p100',
                               'configuration': configuration, 
                               'python_cmd': python_cmd}
        
        params['batchjob_parameters'] = batchjob_parameters
    
        print(batchjob_parameters)
    
    return params, test_dir, data_dir
    
    
def test_general_case(classifier, option, version, cluster_name):
    
    """
    General case:
        - custom metrics
        - special metrics
        - correct layer_threshold, block_threshold
        - invariant before and after save and loading
        - invariant in predict
    """
    
    model = Model()
    params = model.get_default_parameters()  
    params, test_dir, data_dir = setup_parameters(params, classifier, option, version, cluster_name)
    params['special_metrics'] = None
    
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir)
    
    performance, p_history, f_history = model.fit(params,
                                                  data_loading_func,
                                                  [x_train, y_train],
                                                  data_loading_func,
                                                  [x_val, y_val],
                                                  data_loading_func,
                                                  [x_test, y_test],
                                                  True)

    params['convergence_measure'] = 'acc'
    params['metrics'] = ['acc',]
    params['direction'] = 'higher'    
    performance, p_history, f_history = model.fit(params,
                                                  data_loading_func,
                                                  [x_train, y_train],
                                                  data_loading_func,
                                                  [x_val, y_val],
                                                  data_loading_func,
                                                  [x_test, y_test],
                                                  True)

    
    test_performance1 = model.evaluate(data_loading_func, 
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    y1 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    model.save(os.path.join(test_dir,'model.pickle'))
    
    model = Model()
    model.load(os.path.join(test_dir,'model.pickle'))
    
    test_performance2 = model.evaluate(data_loading_func,
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    
    y2 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    for key in test_performance1.keys():
        assert np.allclose(test_performance1[key], test_performance2[key], 1e-5, 1e-5), 'test_performance1=%s \n test_performance2=%s\n' %(str(test_performance1[key]), str(test_performance2[key]))
    assert np.allclose(y1, y2, 1e-5, 1e-5), 'sum(y1)=%.5f \n sum(y2)=%.5f\n' %(np.sum(y1.flatten()), np.sum(y2.flatten()))
    
    
    model.finetune(params,
                   data_loading_func,
                   [x_train, y_train],
                   None,
                   None,
                   None,
                   None)
    
    
    shutil.rmtree(test_dir)
    
    return

def test_custom_loss_case(classifier, option, version, cluster_name):
    
    """
    Custom loss:
        - same settings as general case and using custom loss
    """
    
    model = Model()
    params = model.get_default_parameters()  
    params, test_dir, data_dir = setup_parameters(params, classifier, option, version, cluster_name)
        
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir)

    
    params['loss'] = utils.tf_mae
    performance, p_history, f_history = model.fit(params,
                                                  data_loading_func,
                                                  [x_train, y_train],
                                                  data_loading_func,
                                                  [x_val, y_val],
                                                  data_loading_func,
                                                  [x_test, y_test],
                                                  True)
    
    test_performance1 = model.evaluate(data_loading_func, 
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    y1 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    model.save(os.path.join(test_dir,'model.pickle'))
    
    model = Model()
    model.load(os.path.join(test_dir,'model.pickle'))
    
    test_performance2 = model.evaluate(data_loading_func,
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    
    y2 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    for key in test_performance1.keys():
        assert np.allclose(test_performance1[key], test_performance2[key], 1e-5, 1e-5), 'test_performance1=%s \n test_performance2=%s\n' %(str(test_performance1[key]), str(test_performance2[key]))
    assert np.allclose(y1, y2, 1e-5, 1e-5), 'sum(y1)=%.5f \n sum(y2)=%.5f\n' %(np.sum(y1.flatten()), np.sum(y2.flatten()))
    
    shutil.rmtree(test_dir)

    return

def test_block_threshold_case(classifier, option, version, cluster_name):

    """
    Test growing full blocks for each layers using nonsense block_threshold:
    
    """    
    if classifier in ['popfast', 'popmemo', 'popmemh','pop']:
        return
    
    
    model = Model()
    params = model.get_default_parameters()  
    params, test_dir, data_dir = setup_parameters(params, classifier, option, version, cluster_name)
        
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir)

    
    params['metrics'] = ['mae','mse', utils.tf_mae, 'acc']
    params['convergence_measure'] = 'acc'
    params['block_threshold'] = -1e8
    params['loss'] = utils.tf_mae

    performance, p_history, f_history = model.fit(params,
                                                  data_loading_func,
                                                  [x_train, y_train],
                                                  data_loading_func,
                                                  [x_val, y_val],
                                                  data_loading_func,
                                                  [x_test, y_test],
                                                  True)
    
    for idx in range(1, len(model.model_data['topology'])-1):
        assert len(model.model_data['topology'][idx]) == 4, '#blocks should be 4, only %d from the learned network' % len(model.model_data['topology'][idx])
    
    test_performance1 = model.evaluate(data_loading_func, 
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    y1 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    model.save(os.path.join(test_dir,'model.pickle'))
    
    model = Model()
    model.load(os.path.join(test_dir,'model.pickle'))
    
    test_performance2 = model.evaluate(data_loading_func,
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    
    y2 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    for key in test_performance1.keys():
        assert np.allclose(test_performance1[key], test_performance2[key], 1e-5, 1e-5), 'test_performance1=%s \n test_performance2=%s\n' %(str(test_performance1[key]), str(test_performance2[key]))
    assert np.allclose(y1, y2, 1e-5, 1e-5), 'sum(y1)=%.5f \n sum(y2)=%.5f\n' %(np.sum(y1.flatten()), np.sum(y2.flatten()))
    
    shutil.rmtree(test_dir)
    
    return

def test_layer_threshold_case(classifier, option, version, cluster_name):
  
    """
    Test growing full layers by using nonsense layer_threshold
    """
    
    model = Model()
    params = model.get_default_parameters()  
    params, test_dir, data_dir = setup_parameters(params, classifier, option, version, cluster_name)
        
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir)

    params['metrics'] = ['mae','mse', utils.tf_mae, 'acc']
    params['convergence_measure'] = 'acc'
    params['layer_threshold'] = -1e8
    params['loss'] = utils.tf_mae

    performance, p_history, f_history = model.fit(params,
                                                  data_loading_func,
                                                  [x_train, y_train],
                                                  data_loading_func,
                                                  [x_val, y_val],
                                                  data_loading_func,
                                                  [x_test, y_test],
                                                  True)
    
    assert len(model.model_data['topology']) == 6, '#layer should be 4, given %d from the learned model' % len(model.model_data['topology'])
    
    test_performance1 = model.evaluate(data_loading_func, 
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    y1 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    model.save(os.path.join(test_dir,'model.pickle'))
    
    model = Model()
    model.load(os.path.join(test_dir,'model.pickle'))
    
    test_performance2 = model.evaluate(data_loading_func,
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    
    y2 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    for key in test_performance1.keys():
        assert np.allclose(test_performance1[key], test_performance2[key], 1e-5, 1e-5), 'test_performance1=%s \n test_performance2=%s\n' %(str(test_performance1[key]), str(test_performance2[key]))
    assert np.allclose(y1, y2, 1e-5, 1e-5), 'sum(y1)=%.5f \n sum(y2)=%.5f\n' %(np.sum(y1.flatten()), np.sum(y2.flatten()))
    
    shutil.rmtree(test_dir)
    
    return

def test_both_threshold_case(classifier, option, version, cluster_name): 
    
    """
    Test growing full topology (block and layer) by using nonsense layer and block thresholds
    """
    
    if classifier in ['popfast', 'popmemo', 'popmemh', 'pop']:
        return
    
    model = Model()
    params = model.get_default_parameters()  
    params, test_dir, data_dir = setup_parameters(params, classifier, option, version, cluster_name)
        
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir)

    params['metrics'] = ['mae','mse', utils.tf_mae, 'acc']
    params['convergence_measure'] = 'acc'
    params['layer_threshold'] = -1e8
    params['block_threshold'] = -1e8
    params['loss'] = utils.tf_mae
    
    performance, p_history, f_history = model.fit(params,
                                                  data_loading_func,
                                                  [x_train, y_train],
                                                  data_loading_func,
                                                  [x_val, y_val],
                                                  data_loading_func,
                                                  [x_test, y_test],
                                                  True)
    
    assert len(model.model_data['topology']) == 6, '#layer should be 4, given %d from the learned model' % len(model.model_data['topology'])
    for idx in range(1, len(model.model_data['topology'])-1):
        assert len(model.model_data['topology'][idx]) == 4, '#blocks should be 4, only %d from the learned network' % len(model.model_data['topology'][idx])
    
    test_performance1 = model.evaluate(data_loading_func, 
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    y1 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    model.save(os.path.join(test_dir,'model.pickle'))
    
    model = Model()
    model.load(os.path.join(test_dir,'model.pickle'))
    
    test_performance2 = model.evaluate(data_loading_func,
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    
    y2 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    for key in test_performance1.keys():
        assert np.allclose(test_performance1[key], test_performance2[key], 1e-5, 1e-5), 'test_performance1=%s \n test_performance2=%s\n' %(str(test_performance1[key]), str(test_performance2[key]))
    assert np.allclose(y1, y2, 1e-5, 1e-5), 'sum(y1)=%.5f \n sum(y2)=%.5f\n' %(np.sum(y1.flatten()), np.sum(y2.flatten()))
    
    shutil.rmtree(test_dir)
    
    return

def test_regularizer_case(classifier, option, version, cluster_name):

    """
    Test weight decay during progressive learning and max_norm constraint during finetuning
    """
    
    model = Model()
    params = model.get_default_parameters()  
    params, test_dir, data_dir = setup_parameters(params, classifier, option, version, cluster_name)
        
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir)
    
    params['weight_regularizer'] = 1e-4
    params['weight_constraint'] = None
    params['weight_regularizer_finetune'] = None
    params['weight_constraint_finetune'] = 2.0
    
    performance, p_history, f_history = model.fit(params,
                                                  data_loading_func,
                                                  [x_train, y_train],
                                                  data_loading_func,
                                                  [x_val, y_val],
                                                  data_loading_func,
                                                  [x_test, y_test],
                                                  True)
    
    test_performance1 = model.evaluate(data_loading_func, 
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    y1 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    model.save(os.path.join(test_dir,'model.pickle'))
    
    model = Model()
    model.load(os.path.join(test_dir,'model.pickle'))
    
    test_performance2 = model.evaluate(data_loading_func,
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    
    y2 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    for key in test_performance1.keys():
        assert np.allclose(test_performance1[key], test_performance2[key], 1e-5, 1e-5), 'test_performance1=%s \n test_performance2=%s\n' %(str(test_performance1[key]), str(test_performance2[key]))
    assert np.allclose(y1, y2, 1e-5, 1e-5), 'sum(y1)=%.5f \n sum(y2)=%.5f\n' %(np.sum(y1.flatten()), np.sum(y2.flatten()))
    
    model.finetune(params,
                   data_loading_func,
                   [x_train, y_train],
                   None,
                   None,
                   None,
                   None)
    
    
    shutil.rmtree(test_dir)
    
    return    

def test_input_dropout_case(classifier, option, version, cluster_name):
    
    """
    Test using input_dropout option without using dropout
    """

    model = Model()
    params = model.get_default_parameters()  
    params, test_dir, data_dir = setup_parameters(params, classifier, option, version, cluster_name)
        
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir)
    
    params['input_dropout'] = 0.2
    params['dropout'] = None
    params['dropout_finetune'] = 0.2
    
    performance, p_history, f_history = model.fit(params,
                                                  data_loading_func,
                                                  [x_train, y_train],
                                                  data_loading_func,
                                                  [x_val, y_val],
                                                  data_loading_func,
                                                  [x_test, y_test],
                                                  True)
    
    test_performance1 = model.evaluate(data_loading_func, 
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    y1 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    model.save(os.path.join(test_dir,'model.pickle'))
    
    model = Model()
    model.load(os.path.join(test_dir,'model.pickle'))
    
    test_performance2 = model.evaluate(data_loading_func,
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    
    y2 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    for key in test_performance1.keys():
        assert np.allclose(test_performance1[key], test_performance2[key], 1e-5, 1e-5), 'test_performance1=%s \n test_performance2=%s\n' %(str(test_performance1[key]), str(test_performance2[key]))
    assert np.allclose(y1, y2, 1e-5, 1e-5), 'sum(y1)=%.5f \n sum(y2)=%.5f\n' %(np.sum(y1.flatten()), np.sum(y2.flatten()))
    
    model.finetune(params,
                   data_loading_func,
                   [x_train, y_train],
                   None,
                   None,
                   None,
                   None)
    
    
    shutil.rmtree(test_dir)
    
    return

def test_memory_case(classifier, option, version, mem_type, cluster_name):
    
    """
    Test POPmemO and POPmemH with given memory type ['LDA' or 'PCA']
    """
    
    if classifier in ['hemlgop', 'homlgop', 'hemlrn', 'homlrn', 'popfast', 'pop']:
        return
    
    
    model = Model()
    params = model.get_default_parameters()  
    params, test_dir, data_dir = setup_parameters(params, classifier, option, version, cluster_name)
        
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir)
    
    params['memory_type'] = mem_type
    
    performance, p_history, f_history = model.fit(params,
                                                  data_loading_func,
                                                  [x_train, y_train],
                                                  data_loading_func,
                                                  [x_val, y_val],
                                                  data_loading_func,
                                                  [x_test, y_test],
                                                  True)
    
    test_performance1 = model.evaluate(data_loading_func, 
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    y1 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    model.save(os.path.join(test_dir,'model.pickle'))
    
    model = Model()
    model.load(os.path.join(test_dir,'model.pickle'))
    
    test_performance2 = model.evaluate(data_loading_func,
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    
    y2 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    for key in test_performance1.keys():
        assert np.allclose(test_performance1[key], test_performance2[key], 1e-5, 1e-5), 'test_performance1=%s \n test_performance2=%s\n' %(str(test_performance1[key]), str(test_performance2[key]))
    assert np.allclose(y1, y2, 1e-5, 1e-5), 'sum(y1)=%.5f \n sum(y2)=%.5f\n' %(np.sum(y1.flatten()), np.sum(y2.flatten()))
    
    
    model.finetune(params,
                   data_loading_func,
                   [x_train, y_train],
                   None,
                   None,
                   None,
                   None)
    
    
    shutil.rmtree(test_dir)
    
    return

def test_custom_operator_case(classifier, option, version, cluster_name):
    
    """Test custom operator sets given as callable
    

    """
    def custom_nodal(x,w):
        return tf.cos(x*w)
    
    def custom_pool(x):
        # shape(x): None x I x O
        return tf.reduce_min(x, axis=1)

    def custom_activation(x):
        return tf.nn.relu(-x)
    
    model = Model()
    params = model.get_default_parameters()  
    params, test_dir, data_dir = setup_parameters(params, classifier, option, version, cluster_name)
        
    params['nodal_set'].append(custom_nodal)
    params['pool_set'].append(custom_pool)
    params['activation_set'].append(custom_activation)
    
    
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(data_dir)
    
    
    
    performance, p_history, f_history = model.fit(params,
                                                  data_loading_func,
                                                  [x_train, y_train],
                                                  data_loading_func,
                                                  [x_val, y_val],
                                                  data_loading_func,
                                                  [x_test, y_test],
                                                  True)
    
    test_performance1 = model.evaluate(data_loading_func, 
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    y1 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    model.save(os.path.join(test_dir,'model.pickle'))
    
    model = Model()
    model.load(os.path.join(test_dir,'model.pickle'))
    
    test_performance2 = model.evaluate(data_loading_func,
                                       [x_test, y_test],
                                       params['metrics'],
                                       params['special_metrics'],
                                       params['finetune_computation'])
    
    y2 = model.predict(predict_func,
                       x_test,
                       params['finetune_computation'])
    
    for key in test_performance1.keys():
        assert np.allclose(test_performance1[key], test_performance2[key], 1e-5, 1e-5), 'test_performance1=%s \n test_performance2=%s\n' %(str(test_performance1[key]), str(test_performance2[key]))
    assert np.allclose(y1, y2, 1e-5, 1e-5), 'sum(y1)=%.5f \n sum(y2)=%.5f\n' %(np.sum(y1.flatten()), np.sum(y2.flatten()))
    
    
    model.finetune(params,
                   data_loading_func,
                   [x_train, y_train],
                   None,
                   None,
                   None,
                   None)
    
    
    shutil.rmtree(test_dir)
    
    return


def main(argv):

    try:
      opts, args = getopt.getopt(argv,"s:m:i:v:n:")
    except getopt.GetoptError:
        print('test_models.py  -s <test library from source?> -m <model type> -i <test index> -v <vesion of python> -n <cluster name>')
        sys.exit(2)
    
    is_cluster = False
    for opt, arg in opts:
        if opt == '-n':
            cluster_name = arg
            is_cluster = True
        if opt == '-i':
            test_index = int(arg)
        if opt == '-m':
            classifier = arg
        if opt == '-v':
            version = arg
        if opt == '-s':
            from_source = bool(arg)


    if not is_cluster:
        cluster_name = ''

            
    """
    index 0 -> local cpu
    index 1 -> cluster cpu
    index 2 -> cluster gpu
    index 3 -> local gpu
    """
    global Model

    if from_source:
        cwd = os.getcwd()

        lib_dir = os.path.dirname(os.path.dirname(cwd))

        if not lib_dir in sys.path:
            sys.path = [lib_dir,] + sys.path

    from GOP import models
    
    if classifier == 'hemlgop':
        Model = models.HeMLGOP
    elif classifier == 'homlgop':
        Model = models.HoMLGOP
    elif classifier == 'hemlrn':
        Model = models.HeMLRN
    elif classifier == 'homlrn':
        Model = models.HoMLRN
    elif classifier == 'popfast':
        Model = models.POPfast
    elif classifier == 'popmemo':
        Model = models.POPmemO
    elif classifier == 'popmemh':
        Model = models.POPmemH
    elif classifier == 'pop':
        Model = models.POP
    else:    
        print(classifier)
        raise Exception('Model not recognised')
    
    
    print('???????????? test general case ??????????????????')
    test_general_case(classifier, test_index, version, cluster_name)    
    print('???????????? test custom loss case ??????????????????')
    #test_custom_loss_case(classifier, test_index, version, cluster_name)
    print('???????????? test block threshold case ??????????????????')
    #test_block_threshold_case(classifier, test_index, version, cluster_name)
    print('???????????? test layer threshold case ??????????????????')
    #test_layer_threshold_case(classifier, test_index, version, cluster_name)
    print('???????????? test both threshold case ??????????????????')
    #test_both_threshold_case(classifier, test_index, version, cluster_name)
    print('???????????? test regularizer case ??????????????????')
    #test_regularizer_case(classifier, test_index, version, cluster_name)
    print('???????????? test input dropout case ??????????????????')
    #test_input_dropout_case(classifier, test_index, version, cluster_name)
    print('???????????? test memory case using LDA ??????????????????')
    #test_memory_case(classifier, test_index, version, 'LDA', cluster_name)
    print('???????????? test memory case using PCA ??????????????????')
    #test_memory_case(classifier, test_index, version, 'PCA', cluster_name)
    print('???????????? test custom operator set case ??????????????????')
    #test_custom_operator_case(classifier, test_index, version, cluster_name)
    
    computations = ['local CPU', 'cluster CPU', 'cluster GPU', 'local GPU']
    print('finish testing %s with %s using python %s' % (classifier, computations[test_index], version))

if __name__ == "__main__":
    main(sys.argv[1:])
