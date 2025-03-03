#Code adapted from github.com/estherrolf/representation-matters

import numpy as np
import dill
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from format_data import retrieve_splits
from clusters import retrieve_clusters, retrieve_all_clusters

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
    subset_idxs = shuffled_idxs[:size]
    return take_subset(subset_idxs, *datasets)

'''
Take subset of all instances of group
'''
def take_all_group(group, group_ids, *datasets):
    idxs_this_group = np.where(group_ids == group)[0]
    return take_subset(idxs_this_group, *datasets)

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

'''
Split into pilot, additional train, and test data
'''
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

    pilot_data = {key: None for key in ['X', 'y', 'latlon', 'ids', 'clusters']}
    add_data = {key: None for key in ['X', 'y', 'latlon', 'ids', 'clusters']}
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
        X_pilot, y_pilot, latlon_pilot, ids_pilot, clusters_pilot = take_subset(pilot_idxs_this, 
                                                                X_train, 
                                                                y_train, 
                                                                latlon_train, 
                                                                ids_train,
                                                                groups)
        pilot_data = add_new_data(pilot_data, 
                                  ['X', 'y', 'latlon', 'ids', 'clusters'], 
                                  [X_pilot, y_pilot, latlon_pilot, ids_pilot, clusters_pilot] )
        
        #Additional data
        X_add, y_add, latlon_add, ids_add, clusters_add = take_subset(additional_idxs_this, 
                                                        X_train, 
                                                        y_train, 
                                                        latlon_train, 
                                                        ids_train,
                                                        groups)
        add_data = add_new_data(add_data, 
                                ['X', 'y', 'latlon', 'ids', 'clusters'], 
                                [X_add, y_add, latlon_add, ids_add, clusters_add] )

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

def modified_ipl_logged(ns_stacked, sigmaj_sq, pj, tauj_sq, qj, deltaj):
    nj = ns_stacked[0]
    n = ns_stacked[1]
    
    
    return np.log(deltaj + sigmaj_sq * np.exp(-pj*np.log(nj)) + tauj_sq * np.exp(-qj*np.log(n)))

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
                   fit_logged=True):

    popts = []
    pcovs = []
    for g, group in enumerate(groups):
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

def group_loss_on_pilot(seed,
                        labels,
                        group_sizes_pilot):
    for label in labels:
        pilot_data, add_data, test_data = split_pilot_additional(seed,
                                                                label,
                                                                group_sizes_pilot,
                                                                verbose = True)

        X_pilot_train = pilot_data['X']
        y_pilot_train = pilot_data['y']
        clusters_pilot_train  = pilot_data['clusters']

        _, X_pilot_test, _, y_pilot_test, _, ids_pilot_test, _, clusters_pilot_test = train_test_split(
            add_data['X'], add_data['y'], add_data['ids'], add_data['clusters'], test_size=0.2, random_state=42
        )

        training_set_size = group_sizes_pilot

        # Create a base column
        base_column = np.array([0.01, 0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.21])
        column2 = np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0])
        column3 = np.array([1.0, 0.0, 0, 0, 0, 0, 0, 0])

        # Generate permutations of the base column
        allocations = np.array([np.roll(base_column, shift=i) for i in range(5)])
        allocations = np.vstack([allocations, np.unique([np.random.permutation(column2) for _ in range(1000)], axis=0)])
        allocations = np.vstack([allocations, np.unique([np.random.permutation(column3) for _ in range(100)], axis=0)])
        allocations = allocations.T
        subset_sizes = np.round(allocations*training_set_size).astype(int)
        rmse = np.zeros(subset_sizes.shape)

        alphas = np.logspace(-4, 4, 10)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),     # Step 1: Standardize features
            ('ridgecv', RidgeCV(alphas=alphas, scoring='r2', cv=kf))  # Step 2: RidgeCV with 5-fold CV
        ])

        for i in range(subset_sizes.shape[1]):
            X_train = None
            y_train = None
            #Get specified sizes of each group
            for j in range(subset_sizes.shape[0]):
                size = subset_sizes[j][i]
                print(f"Obtaining subset of cluster {j} of size {size}...")
                X_train_group, y_train_group = take_group_subset(seed, 
                                                                j, 
                                                                clusters_pilot_train, 
                                                                size, 
                                                                X_pilot_train,
                                                                y_pilot_train)
                if X_train is None:
                    X_train = X_train_group
                    y_train = y_train_group
                else:
                    X_train = np.vstack([X_train, X_train_group])
                    y_train = np.concatenate([y_train, y_train_group])

            #Fit the pipeline
            print("Running regression on pilot data...")
            pipeline.fit(X_pilot_train, y_pilot_train)

            #Test on each group
            for j in range(subset_sizes.shape[0]):
                print(f"Testing regression on group {j}")
                X_test_group, y_test_group = take_all_group(j, 
                                                            clusters_pilot_test,
                                                            X_pilot_test,
                                                            y_pilot_test)
                y_pred_group = pipeline.predict(X_test_group)
                rmse[j][i] = mean_squared_error(y_test_group, y_pred_group, squared=False)
            
        write_rmse_and_sizes(label, rmse, subset_sizes)

'''
Write rmse to file
'''
def write_rmse_and_sizes(label, rmse, subset_sizes):
    rmse_and_sizes = {
        'sizes': subset_sizes,
        'RMSE': rmse
    }

    with open(f"data/group_losses/{label}_RMSE.pkl", "wb") as f:
        dill.dump(rmse_and_sizes, f)

if __name__ == '__main__':
    cluster_path = "data/clusters/NLCD_percentages_cluster_assignment.pkl"
    groups = np.unique(retrieve_all_clusters(cluster_path))

    group_loss_on_pilot(42, ["population", "elevation", "treecover"], 1000)
    