from math import pi, log, sqrt
from cvxopt import blas, lapack, solvers
from cvxopt import matrix, spmatrix, spdiag, mul, cos, sin
import  numpy as np
import scipy.linalg

'''Code from Esther Rolf'''

'''
    V-optimality: minimizes the average predicition variance
    Leverage score sampling leads to a good approximation of minimizing the V-optimality criterion 
'''
def v_optimal_design(X):
    print("Determining leverage scores...")
    # full_svd_matrices needs to be False
    full_svd_matrices = False
    #Returns U = unitary matrix with left singular vectors as columns (MxK), s = singular values, Vt = unitary matrix with right singular vectors as rows(KxN)
    U, s, Vt  = scipy.linalg.svd(X, full_matrices = full_svd_matrices)
    
    # compute leverage scores
    l = np.linalg.norm(U, axis=1) ** 2
    
    return l 

#Returns a subset of a matrix of specified size
def sampling(X, size, rule):
    scores = rule(X)

    #Highest scores according to rule
    best_indices = np.argpartition(scores, -size)[-size:]

    print(f"Returning samples according {rule}...")
    return best_indices

def e_optimal_design(V):
    # taken from taken from https://cvxopt.org/examples/book/expdesign.html
    V = matrix(V)
    
    n = V.size[1]
    d = V.size[0]
    G = spmatrix(-1.0, range(n), range(n))
    h = matrix(0.0, (n,1))
    b = matrix(1.0)

    novars = n+1
    # the last element of x is t
    c = matrix(0.0, (novars,1))
    c[-1] = -1.0
    Gs = [matrix(0.0, (d**2,novars))]
    for k in range(n):
        Gs[0][:,k] = -(V[:,k]*V[:,k].T)[:]
        
    # need to make this look like the raveled identity matrix, 
    # so ones on [0, d+1, 2(d+1),..., d**2- 1]
    for j in range(d):
        Gs[0][j*(d+1),-1] = 1.0
        
    hs = [matrix(0.0, (d,d))]
    Ge = matrix(0.0, (n, novars))
    Ge[:,:n] = G
    Ae = matrix(n*[1.0] + [0.0], (1,novars))
    sol = solvers.sdp(c, Ge, h, Gs, hs, Ae, b, solver='mosek')
    xe = sol['x'][:n]
    Z = sol['zs'][0]
    mu = sol['y'][0]
    
    return xe

def e_optimal_design_svd(V, s):
        
    # taken from taken from https://cvxopt.org/examples/book/expdesign.html
    V = matrix(V)
    
    n = V.size[1]
    d = V.size[0]
    G = spmatrix(-1.0, range(n), range(n))
    h = matrix(0.0, (n,1))
    b = matrix(1.0)

    novars = n+1
    # the last element of x is t
    c = matrix(0.0, (novars,1))
    c[-1] = -1.0
    Gs = [matrix(0.0, (d**2,novars))]
    for k in range(n):
        Gs[0][:,k] = -(V[:,k]*V[:,k].T)[:]
        
    # need to make this look like the raveled identity matrix, 
    # so ones on [0, d+1, 2(d+1),..., d**2- 1]
    for j in range(d):
        #Gs[0][j*(d+1),-1] = 1.0
        # replace with s[j]**-2 to use the SVD version
        Gs[0][j*(d+1),-1] = (1.0/s[j])**2
        
    hs = [matrix(0.0, (d,d))]
    Ge = matrix(0.0, (n, novars))
    Ge[:,:n] = G
    Ae = matrix(n*[1.0] + [0.0], (1,novars))
    sol = solvers.sdp(c, Ge, h, Gs, hs, Ae, b, solver='mosek')
    xe = sol['x'][:n]
    Z = sol['zs'][0]
    mu = sol['y'][0]
    
    return xe


def d_optimal_design(V, max_iters=100):
    # modified from https://cvxopt.org/examples/book/expdesign.html
    V = matrix(V)
    n = V.size[1]
    d = V.size[0]
    G = spmatrix(-1.0, range(n), range(n))
    h = matrix(0.0, (n,1))
    A = matrix(1.0, (1,n))
    b = matrix(1.0)


    solvers.options['maxiters'] = max_iters
    # D-design
    #
    # minimize    f(x) = -log det V*diag(x)*V'
    # subject to  x >= 0
    #             sum(x) = 1
    #
    # The gradient and Hessian of f are
    #
    #     gradf = -diag(V' * X^-1 * V)
    #         H = (V' * X^-1 * V)**2.
    #
    # where X = V * diag(x) * V'.

    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0/n, (n,1))
        X = V * spdiag(x) * V.T
        L = +X
        # set L such that X = L * L.T
        try: lapack.potrf(L)
        except ArithmeticError: return None
        # use Cholesky factorization to compute the log det
        f = - 2.0 * sum([log(L[i,i]) for i in range(d)])
        # the following two lines set W = L^{-1}*V
        W = +V
        blas.trsm(L, W)
        # check grad and hessian
        gradf = matrix(-1.0, (1,d)) * W**2
        if z is None: return f, gradf
        H = matrix(0.0, (n,n))
        blas.syrk(W, H, trans='T')
        return f, gradf, z[0] * H**2

    xd = solvers.cp(F, G, h, A = A, b = b)['x']
    
    return xd



def d_optimal_design_with_cost(V, cost_per_instance, total_cost, max_iters=100):
    # modified from https://cvxopt.org/examples/book/expdesign.html
    V = matrix(V)
    n = V.size[1]
    d = V.size[0]
    #G = spmatrix(-1.0, range(n), range(n))
    # add an extra row
    G = spmatrix(-1.0, list(range(n)) + [n]*n, list(range(n)) + list(range(n)) )
    G[n,:] = cost_per_instance
    h = matrix(0.0, (n+1,1))
    h[n] = total_cost
    A = matrix(1.0, (1,n))
    b = matrix(1.0)


    solvers.options['maxiters'] = max_iters
    # D-design
    #
    # minimize    f(x) = -log det V*diag(x)*V'
    # subject to  x >= 0
    #             sum(x) = 1
    #
    # The gradient and Hessian of f are
    #
    #     gradf = -diag(V' * X^-1 * V)
    #         H = (V' * X^-1 * V)**2.
    #
    # where X = V * diag(x) * V'.

    def F(x=None, z=None):
        if x is None: return 0, matrix(1.0/n, (n,1))
        X = V * spdiag(x) * V.T
        L = +X
        # set L such that X = L * L.T
        try: lapack.potrf(L)
        except ArithmeticError: return None
        # use Cholesky factorization to compute the log det
        f = - 2.0 * sum([log(L[i,i]) for i in range(d)])
        # the following two lines set W = L^{-1}*V
        W = +V
        blas.trsm(L, W)
        # check grad and hessian
        gradf = matrix(-1.0, (1,d)) * W**2
        if z is None: return f, gradf
        H = matrix(0.0, (n,n))
        blas.syrk(W, H, trans='T')
        return f, gradf, z[0] * H**2

    xd = solvers.cp(F, G, h, A = A, b = b)['x']
    
    return xd



def a_optimal_design(V):
    #modified from https://cvxopt.org/examples/book/expdesign.html
        
    V = matrix(V)
    d = V.size[0]
    n = V.size[1]
    h = matrix(0.0, (n,1))
    b = matrix(1.0)
    G = spmatrix(-1.0, range(n), range(n))
    
    # A-design.
    #
    # minimize    tr (V*diag(x)*V')^{-1}
    # subject to  x >= 0
    #             sum(x) = 1
    #
    # minimize    tr Y
    # subject to  [ V*diag(x)*V', I ]
    #             [ I,            Y ] >= 0
    #             x >= 0
    #             sum(x) = 1
    novars_in_y = int((d)*(d+1) / 2)
    novars = novars_in_y + n
    
    # pick out the diagonal of y for the trace objective
    c = matrix(0.0, (novars,1))
    # c[[-3, -1]] = 1.0 generalizes to: 
    print('novars = ',novars)
    for i in range(d):
        c[-int(1 + i*d - i*(i-1)/2)] = 1.0
    #Gs = [matrix(0.0, (16, novars))] generalizes to:
    Gs = [matrix(0.0, (4*d**2, novars))]
    from IPython import embed; embed()
    for k in range(n):
#         Gk = matrix(0.0, (4,4))
#         Gk[:2,:2] = -V[:,k] * V[:,k].T
#         Gs[0][:,k] = Gk[:]
        Gk = matrix(0.0, (2*d,2*d)) ## EXCEEDS INT_MAX
        Gk[:d,:d] = -V[:,k] * V[:,k].T
        Gs[0][:,k] = Gk[:]
        
    # put the right values for instantiating the diagonal of y:
#     Gs[0][10,-3] = -1.0
#     Gs[0][11,-2] = -1.0
#     Gs[0][15,-1] = -1.0
    x_variable_index = -1
    for i in range(0,d):
        for j in range(i+1): 
#             print('i = {0}, j= {1}'.format(i,j))
#             print((2*d)**2 - (1 + j + 2*i*d))
            Gs[0][(2*d)**2 - (1 + j + 2*i*d) , x_variable_index] = -1.0
            x_variable_index += -1
    
    #instantate lower left of Gs as I
#     hs = [matrix(0.0, (4,4))]
#     hs[0][2,0] = 1.0
#     hs[0][3,1] = 1.0

    hs = [matrix(0.0, (2*d,2*d))]
    for i in range(d):
        hs[0][d+i,i] = 1.0
     
    Ga = matrix(0.0, (n, novars))
    Ga[:,:n] = G
    # Aa = matrix(n*[1.0] + 3*[0.0], (1,novars)) generalizes to
    Aa = matrix(n*[1.0] + novars_in_y*[0.0], (1,novars)) 
    sol = solvers.sdp(c, Ga, h, Gs, hs, Aa, b)
    xa = sol['x'][:n]
#     Z = sol['zs'][0][:2,:2]
#     mu = sol['y'][0]
    
    return xa