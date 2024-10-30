from math import pi, log, sqrt
#from cvxopt import blas, lapack, solvers
#from cvxopt import matrix, spmatrix, spdiag, mul, cos, sin
import  numpy as np
import scipy.linalg

#V-optimality: minimizes the average predicition variance
def v_optimal_design(X):
    #Leverage score sampling leads to a good approximation of minimizing the V-optimality criterion
    #transpose?, dim MxN

    print("Determining leverage scores...")
    # full_svd_matrices needs to be False
    full_svd_matrices = False
    #Returns U = unitary matrix with left singular vectors as columns (MxK), s = singular values, Vt = unitary matrix with right singular vectors as rows(KxN)
    U, s, Vt  = scipy.linalg.svd(X, full_matrices = full_svd_matrices)
    
    # compute leverage scores
    l = np.linalg.norm(U, axis=1) ** 2
    
    return l

#Returns a subset of a matrix of specified size
def leverage_score_sampling(X, size):
    leverage_scores = v_optimal_design(X)

    #Probability Distribution
    probs = [l/8192 for l in leverage_scores]

    #Subset according to probability distribution
    samples = np.random.choice(X.shape[0], size=size, p=probs, replace=False)

    print("Returning samples according to leverage scores...")
    return samples