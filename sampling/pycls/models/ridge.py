import numpy as np
from sklearn.linear_model import RidgeCV

def ridge_regression():
    model = RidgeCV(alphas=np.logspace(-8,8,15), store_cv_values=False)

    return model