import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold


class RidgeModel:
    def __init__(self, cfg):
        self.cfg = cfg
        pipeline = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())])
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        self.model = GridSearchCV(pipeline, {'ridge__alpha': np.logspace(-5, 5, 10)},
                                   scoring='r2', cv=cv, n_jobs=-1)

    def train(self, X_train, y_train, logger=None):
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test, logger=None):
        r2 = self.model.score(X_test, y_test)
        if logger:
            logger.info(f"R2 score: {r2:.4f}")
        return r2

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
