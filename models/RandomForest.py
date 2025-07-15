import numpy as np

from AppliedML.courselib.models.base import TrainableModel
from AppliedML.courselib.models.tree import DecisionTreeClassifier


class RandomForestClassifier(TrainableModel):
    """
    A Random Forest Classifier that aggregates multiple decision trees and
    uses averaging for regression.
    """

    def __init__(
        self, n_estimators=10, max_depth=5, min_samples_split=2, max_features=None
    ):
        """
        Initializes the Random Forest Classifier.

        Parameters:
        - n_estimators: Number of trees in the forest.
        - max_depth: Maximum depth of each tree.
        - min_samples_split: Minimum number of samples required to split an internal node.
        - max_features: Number of features to consider when looking for the best split.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

        self.trees = []

    def fit(self, X, y):
        """
        Fits the Random Forest Classifier to the training data.

        Parameters:
        - X: Training data features.
        - y: Training data labels.
        """
        num_samples = len(X)

        for _ in range(self.n_estimators):
            indices = np.random.choice(num_samples, size=num_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def decision_function(self, X):
        """
        Computes the decision function for the input data.

        Parameters:
        - X: Input data features.

        Returns:
        - Predicted labels based on the average vote from all trees.
        """
        predictions = np.array([tree(X) for tree in self.trees])
        return self._average_vote(predictions)
        
    def __call__(self, X):
        """
        Calls the decision function to get predictions.

        Parameters:
        - X: Input data features.

        Returns:
        - Predicted labels.
        """
        return self.decision_function(X)

    def _average_vote(self, predictions):
        """
        Averages the predictions from multiple trees.

        Parameters:
        - predictions: Array of predictions from each tree.

        Returns:
        - Averaged predictions.
        """
        labels = np.unique(predictions)
        labels.sort()
        label_ids = {label: i for i, label in enumerate(labels)}

        indices = np.vectorize(lambda x: label_ids[x])(predictions)
        average_index = np.mean(indices, axis=0)

        return labels[np.round(average_index).astype(int)]
