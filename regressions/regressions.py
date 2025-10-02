""" 1. Write function to resave features with splits (train, val, test)
    2. Use Scikit Learn Ridge over range of lambda
    2. Make Sampler function
    3. Make PCA function
    3. Plot regression residuals for each variable of interest """

import dill
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

'''
Run ridge regression and return R2 score
'''
def ridge_regression(X_train, 
                     y_train, 
                     X_test, 
                     y_test, 
                     n_folds=5, 
                     alphas=np.logspace(-5, 5, 10)):
    
    n_samples = X_train.shape[0]

    if n_samples < 2*n_folds:
        print("Not enough samples for cross-validation.")
        return
     
    print("Fitting regression...")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    param_grid = {
        'ridge__alpha': alphas
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    ridge_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='r2',        
        cv=cv,
        n_jobs=-1                   #parallelize across folds
    )


    def evaluate_r2(model, X_test, y_test):
        return model.score(X_test, y_test)
    
    ridge_search.fit(X_train, y_train)
    r2 = evaluate_r2(ridge_search, X_test, y_test)

    if abs(r2) > 1:
        print("Warning: Severe overfitting. Add more samples.")
    return r2
