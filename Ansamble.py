import numpy as np
from DT import Tree

class bagging:
    def __init__(self, base_estimator=None, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators_ = []
    
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.estimators_ = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            estimator = self.base_estimator()
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)
    
    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        return np.mean(predictions, axis=0)
        

class random_forest:
    def __init__(self, n_estimators=10, depth=0, max_depth=None):
        self.n_estimators = n_estimators
        self.depth = depth
        self.max_depth = max_depth
        self.trees = []
    
    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = Tree(depth=self.depth,max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = []
        for i in range(X.shape[0]):
            counts = np.bincount(predictions[:, i])
            final_predictions.append(np.argmax(counts))
        return np.array(final_predictions)



class boosting:
    def __init__(self):
        pass

class stacking:
    def __init__(self):
        pass

