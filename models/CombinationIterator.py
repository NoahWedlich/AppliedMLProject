
class CombinationIterator:
    """
    An iterator that generates combinations of hyperparameters for model tuning.
    """
    
    def __init__(self, parameters, validator=None):
        """
        Initializes the iterator with hyperparameters.
        
        Parameters:
        - parameters: Dictionary from parameter names to their possible values.
        - validator: Optional function to choose valid combinations.
        """
        self.parameters = {
            key: (value if self.is_iterable(value) else [value])
            for key, value in parameters.items()
        }
        
        self.validator = validator
    
    def is_iterable(self, object):
        """
        Check if an object is a true iterable (not a string).
        """
        try:
            iter(object)
            return True
        except TypeError:
            return False
            
    def __iter__(self):
        """
        Iterates through all combinations of hyperparameters.
        
        Yields:
        - Dictionary of hyperparameters for each combination.
        """
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())
        
        for combination in self._generate_combinations(values):
            params = {keys[i]: combination[i] for i in range(len(keys))}
            if self.validator is None or self.validator(params):
                yield params
            
    def _generate_combinations(self, values):
        """
        Generates all combinations of the provided values.
        
        Parameters:
        - values: List of lists, where each inner list contains possible values for a parameter.
        """
        if not values:
            yield []
            return
        
        # For each value in the first list, recursively generate combinations of the tail
        first, *rest = values
        for value in first:
            for combination in self._generate_combinations(rest):
                yield [value] + combination