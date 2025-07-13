import multiprocessing as mp

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
    
    def __init__(self, model_class, hyperparameters, validator=None, process_count=None):
        self.model_class = model_class
        self.combination_iterator = CombinationIterator(hyperparameters, validator)
        
        if process_count is None:
            process_count = mp.cpu_count() // 2
        self.process_count = max(1, process_count)
        
        self.models = []
        
    def get_model(self, parameters):
        optimizer = parameters.get('optimizer', None)
        if callable(optimizer):
            parameters['optimizer'] = optimizer(parameters)
        
        return self.model_class(**parameters)
        
    def _fit_model(self, index, model, X, y, training_params=None):
        metrics = model.fit(X, y, **(training_params or {}))
        print(f"Model {index} fitted.")
        return (metrics, model)
        
    def fit(self, X, y, training_params=None):
        params = list(self.combination_iterator)
        models = [self.get_model(params) for params in params]
        
        if callable(training_params):
            instantiate_training_params = [training_params(param) for param in params]
        else:
            instantiate_training_params = [training_params] * len(params)
        
        arguments = [(i, model, X, y, t_params)
            for i, (model, t_params) in enumerate(zip(models, instantiate_training_params))]
        
        process_count = min(self.process_count, len(models))
        with mp.Pool(process_count) as pool:
            print(f"Started fitting {len(models)} models on {process_count} processes.")
            results = pool.starmap(self._fit_model, arguments)
        
        self.models = [(model, params, metrics) for params, (metrics, model) in zip(params, results)]
        return self.models
