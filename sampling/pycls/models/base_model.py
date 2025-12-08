class BaseModel:
    """Abstract base class for all models."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        
    def train(self, X_train, y_train, X_val, y_val, logger=None):
        """Train the model. Must be implemented by subclasses."""
        raise NotImplementedError
        
    def evaluate(self, X_test, y_test, logger=None):
        """Evaluate the model. Must be implemented by subclasses."""
        raise NotImplementedError
        
    def predict(self, X):
        """Make predictions. Must be implemented by subclasses."""
        raise NotImplementedError
        
    def score(self, X, y):
        """Score the model (for sklearn compatibility)."""
        raise NotImplementedError