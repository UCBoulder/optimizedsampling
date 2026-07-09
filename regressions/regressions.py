import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler


def ridge_regression(X_train, y_train, X_test, y_test, n_folds=5, alphas=np.logspace(-5, 5, 10)):
    if X_train.shape[0] < 2 * n_folds:
        return None

    model = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    search = GridSearchCV(model, {'ridge__alpha': alphas}, scoring='r2', cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search.score(X_test, y_test)
