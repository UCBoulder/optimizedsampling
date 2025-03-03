'''
Adapted from github.com/estherrolf/representation-matters
'''
import pandas as pd
import numpy as np
import sys
import os
import pickle
import argparse
import itertools
import time
import scipy.sparse

import sklearn
import sklearn.metrics

import representation_matters.code.scripts.train_fxns_nonimage as train_fxns_nonimage
from representation_matters.code.scripts.train_fxns_nonimage import subset_and_train

from opt import solve_rep_matters

data_dir = 'representation_matters/data'

def kwargs_to_kwargs_specifier(model_kwargs):
    return '_'.join(['_'.join([str(y) for y in x]) for x in  model_kwargs.items()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, 
                        help='string identifier of the dataset')
    parser.add_argument('--num_seeds', type=int, default=10, 
                        help='number of seeds to run')
    parser.add_argument('--seed_beginning', type=int, default=0, 
                        help='seed to start with')
    parser.add_argument('--reweight', type=bool, default=False, 
                        help='whether to reweight data to the pop proportions')
    parser.add_argument('--pred_fxn_name', type=str, default='rf_classifier', 
                        help='which pred function to run the cv for')
    parser.add_argument('--results_tag', type=str, default='debug', 
                        help='tag for results descriptor')

    args = parser.parse_args()
    num_seeds, seed_beginning = args.num_seeds, args.seed_beginning
    dataset_name = args.dataset_name
    run_type = 'sampling'
    print('run type',run_type)
    pred_fxn_name = args.pred_fxn_name
    print('pred_fxn_name is:', pred_fxn_name)
    results_tag = args.results_tag
    reweight = args.reweight
    print('reweight is:', reweight)
    
    results_general_path = 'subset_results/'
    
    # do ten random seeds
    num_eval_seeds = num_seeds
    seed_start = seed_beginning

    obj_str = 'ERM' #can add importance weighting later if needed

    data = None
    X = None
    label_colname = ''
    group_key = ''
    acc_fxns = {}
    pred_fxn = None
    model_kwargs = {}
    results_descriptor = ''
    reweight_target_dist = []
    
    # default to false
    predict_prob = False
    
    # get parameters for each dataset
    if dataset_name.lower() == 'goodreads':
        label_colname = 'rating'
        
        all_group_colnames = ['history', 'fantasy']
        group_key = 'genre'
        
        data_dir_goodreads = os.path.join(data_dir, 'goodreads')
        data_fn = os.path.join(data_dir_goodreads,
                               'goodreads_{0}_{1}_5_{2}_fold_splits.csv'.format(all_group_colnames[0],
                                                                           all_group_colnames[1],
                                                                               group_key))
        
        features_fn =  data_fn.replace('5_{0}_fold_splits.csv'.format(group_key), 'features_2k.npz')
        
        data = pd.read_csv(data_fn)
        X = scipy.sparse.load_npz(features_fn).toarray()
        
        # set up the acc_keys
        acc_fxns = {'mse': sklearn.metrics.mean_squared_error,
                    'mae': sklearn.metrics.mean_absolute_error}
          
        pred_fxn_name = 'logistic_regression'
        pred_fxn = train_fxns_nonimage.fit_logistic_regression_multiclass
        
        if pred_fxn_name == 'logistic_regression':
            model_kwargs = {'penalty': 'l2','C':1.0, 'solver':'lbfgs'}
        elif pred_fxn_name == 'ridge':
            if reweight:
                def model_kwargs_by_alphas(alphas_both):
                    alpha_thresh = 0.05
                    if np.min(alphas_both) <= alpha_thresh:
                        return {'alpha': 1.0}
                    else:
                        return {'alpha': 0.1}
            else:
                model_kwargs = {'alpha': 0.1}
            
               
        results_descriptor = 'goodreads_2k_{1}_{2}_{0}_'.format(run_type,
                                                                all_group_colnames[0],
                                                                all_group_colnames[1])+obj_str
        
        if reweight:
            gamma = dataset_params[dataset_name.lower()]['gamma']
            reweight_target_dist = [gamma,1-gamma]
            print('reweighting to ',reweight_target_dist)
            
        # don't use this since params can vary by allocation
        if reweight:
            kwargs_specifier = 'subset'
        else:
            kwargs_specifier = kwargs_to_kwargs_specifier(model_kwargs)
    else:
        print('TODO: need to input the data files for {0}'.format(dataset_name))

    fold_key = 'fold'
    train_data = data[data[fold_key] == 'train']

    total_n_train = len(train_data)
    max_group_sizes = train_data.groupby(group_key).count().iloc[:,0].values

    groups = all_group_colnames
    group_costs = np.array([1, 10])

    budget = 50000 #make argument
    group_sample_sizes = solve_rep_matters(groups,
                                           max_group_sizes,
                                           group_costs, 
                                           budget,
                                           prop=True, #use gammajs
                                           l=0.01) #make argument
    group_sample_sizes = np.vstack(np.round(group_sample_sizes).astype(int)) #change to probabilistic sampling
    print(group_sample_sizes)

    # instantiate fps where output data is to be stored
    this_results_path = os.path.join(results_general_path, results_descriptor)

    # if output dir doesn't exist for this pred function, create it
    pred_fxn_base_name = 'subset_{0}'.format(group_key)
    this_results_path_pre = os.path.join(this_results_path,pred_fxn_base_name)
    results_path_this_pred_fxn = os.path.join(this_results_path_pre, pred_fxn_name)
    
    for p in [results_general_path,this_results_path,this_results_path_pre,results_path_this_pred_fxn]:
        if not os.path.exists(p):
            print(p,' did not exist; making it now')
            os.makedirs(p)

    for s in range(seed_start,seed_start+num_eval_seeds):
        t1 = time.time()
        print('seed: ',s)
        results = subset_and_train(data, 
                                  X, 
                                  group_key, 
                                  label_colname, 
                                  group_sample_sizes, 
                                  pred_fxn = pred_fxn,
                                  model_kwargs = model_kwargs,
                                  acc_fxns = acc_fxns,
                                  reweight = reweight,
                                  reweight_target_dist = reweight_target_dist,
                                  fold_key = 'fold',
                                  eval_key = 'test',
                                  seed_start = s,
                                  predict_prob = predict_prob,
                                  num_seeds = 1,
                                  verbose=False)
                        
        accs_by_group, accs_total = results 

        # aggregate results for saving
        results_dict = {'group_key': group_key,
                        'seed_start': seed_start,
                        'num_seeds': num_eval_seeds,
                        'subset_sizes': group_sample_sizes,
                        'accs_by_group': accs_by_group, 
                        'accs_total': accs_total}

        # save the eval results for this hyperparameter setting
        fn_save = os.path.join(results_path_this_pred_fxn,kwargs_specifier+'_seed_{0}.pkl'.format(s))
        print('saving overall results results in ', fn_save)

        with open(fn_save, 'wb') as f:
            pickle.dump(results_dict, f)
            
        t2 = time.time()    
        print('seed {0} finished in {1} minutes'.format(s, (t2-t1)/60))