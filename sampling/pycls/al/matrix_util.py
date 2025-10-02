import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def get_sparse_farthest_neighbors(distance_matrix, n_neighbors=10):
    n = distance_matrix.shape[0]
    #sort indices
    sorted_indices = np.argsort(-distance_matrix, axis=1)  # minus for descending
    sorted_distances = -np.sort(-distance_matrix, axis=1)  # descending distances

    #take top n_neighbors farthest
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = sorted_indices[:, :n_neighbors].flatten()
    data = sorted_distances[:, :n_neighbors].flatten()

    sparse_dist = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return sparse_dist