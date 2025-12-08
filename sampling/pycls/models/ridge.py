from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from .base_model import BaseModel

class RidgeModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = self._build_pipeline()
        
    def _build_pipeline(self):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        
        param_grid = {
            'ridge__alpha': np.logspace(-5, 5, 10)
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='r2',
            cv=cv,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train, logger=None):
        if logger:
            logger.info("Training ridge regression model...")
        self.model.fit(X_train, y_train)
        return self
        
    def evaluate(self, X_test, y_test, logger=None):
        r2 = self.model.score(X_test, y_test)
        if logger:
            logger.info(f"R² score: {r2:.4f}")
        return r2
        
    def predict(self, X):
        return self.model.predict(X)
        
    def score(self, X, y):
        return self.model.score(X, y)