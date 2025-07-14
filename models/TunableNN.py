
from AppliedML.courselib.models import nn

from TunableModel import TunableModel

class TunableNN(TunableModel):
    """
    A specialized TunableModel for Neural Networks that allows tuning of hyperparameters
    """
    
    def __init__(self, hyperparameters, validator):
        """
        Initializes the TunableNN with hyperparameters.
        
        Parameters:
        - hyperparameters: Dictionary of hyperparameters to tune.
        - validator: Function to select valid combinations of hyperparameters.
        """
        if 'widths' not in hyperparameters:
            raise ValueError("Widths must be specified in hyperparameters.")
            
        if 'optimizer' not in hyperparameters:
            raise ValueError("Optimizer must be specified in hyperparameters.")
            
        super().__init__(
            model_class=nn.MLP,
            hyperparameters=hyperparameters,
            validator=validator,
        )