#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This is an example showing how to use the model given in the PyGOP with
a subset of CelebA dataset


Author: Dat Tran
Email: dat.tranthanh@tut.fi, viebboy@gmail.com
github: https://github.com/viebboy
"""

import data_utility, os, time, six, getopt, sys
from GOP import models


def main(argv):

    try:
      opts, args = getopt.getopt(argv,"m:c:")
    except getopt.GetoptError:
        print('train_miniCelebA.py -m <model> -c <computation option cpu/gpu>')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-m':
            model_name = arg
        if opt == '-c':
            computation = arg
            
            
    # input 512 deep features
    # output 20 class probability
    input_dim = 512
    output_dim = 20

        
    if computation == 'cpu':
        search_computation = ('cpu', 8)
        finetune_computation = ('cpu', )
    else:
        search_computation = ('gpu', [0,1,2,3])
        finetune_computation = ('gpu', [0,1,2,3])
        

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
    
    # create POP model
    model = Model()
    model_name += '_miniCelebA'
    
    # get default parameters and assign some specific values
    params = model.get_default_parameters()
    
    tmp_dir = os.path.join(os.getcwd(), 'tmp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    
    params['tmp_dir'] = tmp_dir
    params['model_name'] = model_name 
    params['input_dim'] = input_dim
    params['output_dim'] = output_dim
    params['metrics'] = ['acc',]
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
    params['lr_train'] = (0.01, 0.005, 0.001, 0.0005, 0.0001)
    params['epoch_train'] = (20, 40, 40, 40, 40)
    params['lr_finetune'] = (0.01, 0.005, 0.001, 0.0005, 0.0001)
    params['epoch_finetune'] = (20,40, 40, 40, 40)
    params['direct_computation'] = True
    

    batch_size = 64
    start_time = time.time()
    
    
    train_arguments = ['data/miniCelebA_x_train.npy', 'data/miniCelebA_y_train.npy', batch_size, True]
    val_arguments = ['data/miniCelebA_x_val.npy', 'data/miniCelebA_y_val.npy', batch_size, False]
    test_arguments = ['data/miniCelebA_x_test.npy', 'data/miniCelebA_y_test.npy', batch_size, False]
    data_function = data_utility.load_miniCelebA

    performance, _, _ = model.fit(params,
                                  train_func = data_function,
                                  train_data = train_arguments,
                                  val_func = data_function,
                                  val_data = val_arguments,
                                  test_func = data_function ,
                                  test_data = test_arguments,
                                  verbose=True)
    
    stop_time = time.time()
    
    with open('result.txt','a') as fid:
        fid.write('Finish training %s in %.2f seconds with following performance: \n' %(model_name, stop_time-start_time))
    
        for key in performance.keys():
            if performance[key] is not None:
                fid.write('%s: %.4f \n' %(key, performance[key]))
            

if __name__ == "__main__":
    main(sys.argv[1:])



    
