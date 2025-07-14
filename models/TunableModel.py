import multiprocessing as mp

from models.CombinationIterator import CombinationIterator

class TunableModel:
    """
    Represents a model that can be tuned with various hyperparameters.
    """
    
    def __init__(self, model_class, hyperparameters, validator=None, process_count=None):
        """
        Initializes the TunableModel with a model class and hyperparameters.
        
        Parameters:
        - model_class: Class of the model to be tuned.
        - hyperparameters: Dictionary of hyperparameters to tune.
        - validator: Optional function to select valid combinations of hyperparameters.
        - process_count: Number of processes to use for parallel fitting. Defaults to half the CPU count.
        """
        self.model_class = model_class
        self.combination_iterator = CombinationIterator(hyperparameters, validator)
        
        if process_count is None:
            process_count = mp.cpu_count() // 2
        self.process_count = max(1, process_count)
        
        self.models = []
        
    def get_model(self, parameters):
        """
        Returns an instance of the model with the given parameters.
        
        Parameters:
        - parameters: Dictionary of hyperparameters for the model.
        
        Returns:
        - An instance of the model class initialized with the given parameters.
        """
        optimizer = parameters.get('optimizer', None)
        if callable(optimizer):
            # If the optimizer is a callable, instantiate it with the parameters
            parameters['optimizer'] = optimizer(parameters)
        
        return self.model_class(**parameters)
        
    def _fit_model(self, index, model, X, y, training_params=None):
        """
        Fits the model with the given data and training parameters.
        
        Parameters:
        - index: Index of the model in the list.
        - model: Instance of the model to fit.
        - X: Training data features.
        - y: Training data labels.
        - training_params: Additional parameters for training the model.
        
        Returns:
        - Tuple containing the metrics and the fitted model.
        """
        metrics = model.fit(X, y, **(training_params or {}))
        return (metrics, model)
        
    def fit(self, X, y, training_params=None):
        """
        Fits all models with the given training data and parameters.
        
        Parameters:
        - X: Training data features.
        - y: Training data labels.
        - training_params: Additional parameters for training the models. Can be a callable or a dictionary.
        
        Returns:
        - List of tuples containing the fitted model, their parameters, and the metrics.
        """
        params = list(self.combination_iterator)
        models = [self.get_model(params) for params in params]
        
        # Instantiate training parameters if necessary
        if callable(training_params):
            instantiate_training_params = [training_params(param) for param in params]
        else:
            instantiate_training_params = [training_params] * len(params)
        
        arguments = [(i, model, X, y, t_params)
            for i, (model, t_params) in enumerate(zip(models, instantiate_training_params))]
        
        # Use multiprocessing to fit models in parallel
        process_count = min(self.process_count, len(models))
        with mp.Pool(process_count) as pool:
            results = pool.starmap(self._fit_model, arguments)
        
        self.models = [(model, params, metrics) for params, (metrics, model) in zip(params, results)]
        return self.models
