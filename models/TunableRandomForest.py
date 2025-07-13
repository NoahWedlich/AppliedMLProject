
from models.RandomForest import RandomForestClassifier

from models.TunableModel import TunableModel

class TunableRandomForest(TunableModel):
    
    def __init__(self, hyperparameters, validator=None, random_seed=None):
        if 'n_estimators' not in hyperparameters:
            raise ValueError("Number of estimators must be specified in hyperparameters.")
        
        if 'max_depth' not in hyperparameters:
            raise ValueError("Max depth must be specified in hyperparameters.")
        
        super().__init__(
            model_class=RandomForestClassifier,
            hyperparameters=hyperparameters,
            validator=validator,
            random_seed=random_seed
        )