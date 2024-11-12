import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca(train, test):
    X = np.concatenate((train, test), axis=0)

    #Scale data
    # scaling = StandardScaler()
    # scaling.fit(X)
    # train = scaling.transform(train)
    # test = scaling.transform(test)

    # from IPython import embed; embed()

    #PCA
    print("Performing PCA...")
    pca = PCA(n_components=8192)
    X_pca = pca.fit_transform(X)

    print("Number of PCA Components: ", pca.n_components_)

    #Separate train and test
    train_pca = X_pca[:train.shape[0], :]
    test_pca = X_pca[train.shape[0]:, :]

    from IPython import embed; embed()
    return train_pca, test_pca


# def pca(train, test):
#     #Scale data and perform pca
#     # scaling = StandardScaler()
#     # scaling.fit(train)
#     # train = scaling.transform(train)
#     # test = scaling.transform(test)

#     pca = PCA(0.95)
#     train = pca.fit_transform(train)
#     test = pca.fit_transform(test)
#     print("Number of PCA Components: ", pca.n_components_)

#     return train, test

def to_df(set):
    set_df = pd.DataFrame(set, columns=set.columns, index=set.index)
    return set_df