import numpy as np
import random

class CombinationIterator:
    
    def __init__(self, parameters, validator=None):
        self.parameters = {
            key: (value if self.is_iterable(value) else [value])
            for key, value in parameters.items()
        }
        
        self.validator = validator
    
    def is_iterable(self, object):
        try:
            iter(object)
            return True
        except TypeError:
            return False
            
    def __iter__(self):
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())
        
        for combination in self._generate_combinations(values):
            params = {keys[i]: combination[i] for i in range(len(keys))}
            if self.validator is None or self.validator(params):
                yield params
            
    def _generate_combinations(self, values):
        if not values:
            yield []
            return
        
        first, *rest = values
        for value in first:
            for combination in self._generate_combinations(rest):
                yield [value] + combination

class TunableModel:
    
    def __init__(self, model_class, hyperparameters, validator=None, optimizer=None,):
        self.model_class = model_class
        self.optimizer = optimizer
        self.combination_iterator = CombinationIterator(hyperparameters, validator)
              
        self.seed = random.randint(0, 1000000)
        
        self.models = []
        
    def get_model(self, parameters):
        parameters['optimizer'] = self.optimizer
        np.random.seed(self.seed)
        return self.model_class(**parameters)
        
    def fit(self, X, y, training_params=None, metrics_dict=None):
        for params in self.combination_iterator:
            model = self.get_model(params)
            
            metrics = model.fit(X, y, **(training_params or {}), **(metrics_dict or {}))
            
            self.models.append((model, params, metrics))
            
        return self.models