import numpy as np

from AppliedML.courselib.models.base import TrainableModel
from AppliedML.courselib.models import tree

class RandomForestClassifier(TrainableModel):
    
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features=None, random_seed=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        
        self.random_seed = random_seed
        np.random.seed(random_seed or np.random.randint(0, 1000000))
        
        self.trees = []
        
    def fit(self, X, y):
        num_samples = len(X)
        
        for _ in range(self.n_estimators):
            indices = np.random.choice(num_samples, size=num_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            tree = tree.DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
    def decision_function(self, X):
        predictions = np.array([tree(X) for tree in self.trees])
        return self._average_vote(predictions)
    
    def _average_vote(self, predictions):
        labels = np.unique(predictions)
        label_ids = {label: i for i, label in enumerate(labels)}
        
        indices = np.vectorize(lambda x: labels[x])(predictions)
        average_index = np.mean(indices, axis=0)
        
        return labels[np.round(average_index).astype(int)]