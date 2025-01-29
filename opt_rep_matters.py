#Code adapted from github.com/estherrolf/representation-matters

import numpy as np
from scipy.optimize import curve_fit

from format_data import retrieve_splits
from clusters import retrieve_clusters
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

'''
Take subset with subset idxs of (multiple) dataset(s)
'''
def take_subset(subset_idxs, *datasets):
    for dataset in datasets:
        yield dataset[subset_idxs]

'''
Take subset of specific group
'''
def take_group_subset(seed, group, group_ids, size, *datasets):
    rs = np.random.RandomState(seed)

    idxs_this_group = np.where(group_ids == group)[0]
    shuffled_idxs = rs.choice(len(idxs_this_group), 
                                len(idxs_this_group),
                                replace = False)
    subset_indices = shuffled_idxs[:size]
    return take_subset(subset_idxs, *datasets)

'''
Append new data to data in a dict
'''
def add_new_data(data_dict, keys, new_data):
    # For each key, stack the corresponding data
    for key, data in zip(keys, new_data):
        if data_dict[key] is None:
            # Initialize with the first dataset
            data_dict[key] = data
        else:
            #Stack data
            if data_dict[key].ndim == 1 and data.ndim == 1:
                data_dict[key] = np.concatenate([data_dict[key], data])
            else:
                data_dict[key] = np.vstack([data_dict[key], data])
    return data_dict

def split_pilot_additional(seed,
                           label,
                           group_sizes_pilot, #int, assumes all groups are same size
                           verbose = True):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        latlon_train,
        latlon_test,
        ids_train,
        ids_test
    ) = retrieve_splits(label)

    cluster_path = "data/clusters/NLCD_percentages_cluster_assignment.pkl" #originally as parameter
    groups = retrieve_clusters(ids_train, cluster_path)
    
    rs = np.random.RandomState(seed)

    pilot_data = {key: None for key in ['X', 'y', 'latlon', 'ids']}
    add_data = {key: None for key in ['X', 'y', 'latlon', 'ids']}
    test_data = {
        'X': X_test,
        'y': y_test,
        'latlon': latlon_test,
        'ids': ids_test
    }
    
    for g in np.unique(groups):
        print(f"Determining pilot data for group {g}")
        idxs_this_group = np.where(groups == g)[0]
        shuffled_idxs = rs.choice(len(idxs_this_group), 
                                  len(idxs_this_group),
                                  replace = False)
        
        pilot_idxs_this = shuffled_idxs[:group_sizes_pilot]
        additional_idxs_this = shuffled_idxs[group_sizes_pilot:]
        
        #Pilot data
        X_pilot, y_pilot, latlon_pilot, ids_pilot = take_subset(pilot_idxs_this, X_train, y_train, latlon_train, ids_train)
        pilot_data = add_new_data(pilot_data, 
                                  ['X', 'y', 'latlon', 'ids'], 
                                  [X_pilot, y_pilot, latlon_pilot, ids_pilot] )
        
        #Additional data
        X_add, y_add, latlon_add, ids_add = take_subset(additional_idxs_this, X_train, y_train, latlon_train, ids_train)
        add_data = add_new_data(add_data, 
                                ['X', 'y', 'latlon', 'ids'], 
                                [X_add, y_add, latlon_add, ids_add] )

    # shuffle the data so it isn't sorted by group
    size_of_pilot = pilot_data['X'].shape[0]
    shuffled_pilot_idxs = rs.choice(size_of_pilot, size=size_of_pilot, replace=False)
    for key, data in pilot_data.items():
        pilot_data[key] = data[shuffled_pilot_idxs]

    size_of_add = add_data['X'].shape[0]
    shuffled_add_idxs = rs.choice(size_of_add, size=size_of_add, replace=False)
    for key, data in add_data.items():
        add_data[key] = data[shuffled_add_idxs]

    if verbose:
        print('of the remaining {0} pts: '.format(len(ids_train)), end = ' ') 
        print('{0} pts in pilot, and {1} pts in additional'.format(len(pilot_data['ids']),
                                                               len(add_data['ids'])))
    return pilot_data, add_data, test_data

'''
Modified inverse power law from Representation Matters
'''
def modified_ipl(ns_stacked, sigmaj_sq, pj, tauj_sq, qj, deltaj):
    nj = ns_stacked[0]
    n = ns_stacked[1]
    
    return deltaj + sigmaj_sq * np.exp(-pj*np.log(nj)) + tauj_sq * np.exp(-qj*np.log(n)) 

'''
Fit scaling law from Representation Matters
I.e. determine sigma_g, tau_g, p_g, q_g for each group
'''
def fit_scaling_law(x,y , delta_bounds = [0,np.inf], fit_logged=True, min_pts = 1):
    
    bounds_ipl = np.array([[0,np.inf], [0,2], [0,np.inf], [0,2], delta_bounds]).T
    #bounds_ipl = np.array([[0,np.inf], [0,np.inf], [0,np.inf], [0,np.inf], delta_bounds]).T
    
    # scaling law won't work if there's zero of any data point
    valid_idxs = np.intersect1d(np.where(x[0,:] >=min_pts)[0],
                                np.where(x[1,:] >=min_pts)[0])
    
    # defaults to 1 unless otherwise stated
    initial_guess = [1,1,1,1,1e-8]
    
    if fit_logged:
        popt, pcov = curve_fit(modified_ipl_logged, 
                               x[:,valid_idxs], 
                               np.log(y[valid_idxs]),  
                               p0 = initial_guess,
                               bounds= bounds_ipl, 
                               max_nfev=1000)
        
    else:
        popt, pcov = curve_fit(modified_ipl,
                               x[:,valid_idxs], 
                               y[valid_idxs],  
                               p0 = initial_guess,
                               bounds= bounds_ipl, 
                               max_nfev=1000)
        
    # if we're hitting the bounds on exponents through a message
    if popt[1] == bounds_ipl[0,1] or popt[1] == bounds_ipl[1,1]:
        print('estimated pj hit bound: {0}'.format(popt[1]))
    if popt[3] == bounds_ipl[0,3] or popt[3] == bounds_ipl[1,3]:
        print('estimated qj hit bound: {0}'.format(popt[3]))
    
    return popt, pcov

'''
Obtain data and fit scaling law
'''
def get_group_fits(groups, 
                   accs_by_group, 
                   subset_sizes,
                   min_pts = 1,
                   delta_bounds = [0,np.inf],
                   verbose=True, 
                   fit_logged=True,
                   need_to_tile_data=True):

    popts = []
    pcovs = []
    for g, group in enumerate(groups):
        if need_to_tile_data:
            ns, y = tile_data(subset_sizes.sum(axis=0), 
                              accs_by_group[g])

            njs, _ = tile_data(subset_sizes[g], 
                              accs_by_group[g])
        else:
            ns = subset_sizes.sum(axis=0)
            njs = subset_sizes[g]
            y = accs_by_group[g]

        ns_input = np.vstack((njs, ns))
        popt, pcov = fit_scaling_law(ns_input, y, 
                                     delta_bounds = delta_bounds,
                                     min_pts = min_pts,
                                    fit_logged=fit_logged)

        popts.append(popt)
        pcovs.append(pcov)
        
        if verbose:
            print(group)
            keys = ['sigmaj_sq', 'p', 'tauj_sq', 'q', 'deltaj']
            for k, key in enumerate(keys):
                print("{0} : {1:.3f} ({2:.3f})".format(key,popt[k],np.sqrt(pcov[k,k])))
            print()
    return popts, pcovs

if __name__ == '__main__':
    for label in ["population", "elevation", "treecover"]:
        pilot_data, add_data, test_data = split_pilot_additional(42,
                                                                "population",
                                                                1000,
                                                                verbose = True)

        X_pilot_train, X_pilot_test, y_pilot_train, y_pilot_test, latlon_pilot_train, latlon_pilot_test, ids_pilot_train, ids_pilot_test = train_test_split(
            pilot_data['X_pilot'], pilot_data['y_pilot'], pilot_data['latlon_pilot'], pilot_data['ids_pilot'], test_size=0.2, random_state=42
        )
        cluster_path = "data/clusters/NLCD_percentages_cluster_assignment.pkl" #originally as parameter
        clusters_pilot_train = retrieve_clusters(ids_pilot_train, cluster_path)
        clusters_pilot_test = retrieve_clusters(ids_pilot_test, cluster_path)

        training_set_size = np.min([len(cluster_pilot_train == c) for c in np.unique(cluster_pilot_train)])

        # Create a base column
        base_column = np.array([0.01, 0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.21])
        column2 = np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0])
        column3 = np.array([1.0, 0.0, 0, 0, 0, 0, 0, 0])

        # Generate permutations of the base column
        allocations = np.array([np.roll(base_column, shift=i) for i in range(5)])
        allocations = np.vstack([allocations, np.unique([np.random.permutation(column2) for _ in range(1000)], axis=0)])
        allocations = np.vstack([allocations, np.unique([np.random.permutation(column3) for _ in range(100)], axis=0)])
        allocations = allocations.T
        subset_sizes = allocations*training_set_size

        rmse = np.zeros(subset_sizes.shape)

        alphas=np.logspace(-5, 5, 100)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        #Pipeline that scales and then fits ridge regression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),     # Step 1: Standardize features
            ('ridgecv', RidgeCV(alphas=alphas, scoring='r2', cv=kf))  # Step 2: RidgeCV with 5-fold CV
        ])

       for i in range(subset_sizes.shape[1]):
            for j in range(subset_sizes.shape[0]):
                size = subset_sizes
                subset_indices = np.random.choice(len(X_train), size=n, replace=False)

            #Fit the pipeline
            pipeline.fit(X_train, y_train)


            rmse = mean_squared_error(y_test, y_pred, squared=False)