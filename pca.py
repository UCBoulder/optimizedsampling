import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def pca(train, test):
    #Scale data and perform pca
    scaling = StandardScaler()
    scaling.fit(train)
    train = scaling.transform(train)
    test = scaling.transform(test)

    pca = PCA(0.95)
    pca.fit(train)
    print("Number of PCA Components: ", pca.n_components_)
    train = pca.transform(train)
    test = pca.transform(test)

    return train, test
