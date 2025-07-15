from AppliedML.courselib.models import nn
from AppliedML.courselib.optimizers import GDOptimizer

from models.TunableModel import TunableModel

class TunableNN(TunableModel):
    """
    A specialized TunableModel for Neural Networks that allows tuning of hyperparameters
    """
    
    def get_optimizer(params):
        if params['activation'] == 'ReLU':
            return GDOptimizer(learning_rate=0.5)
        else:
            return GDOptimizer(learning_rate=5)

    def __init__(self, hyperparameters, validator=None, process_count=None, random_seed=None):
        """
        Initializes the TunableNN with hyperparameters.

        Parameters:
        - hyperparameters: Dictionary of hyperparameters to tune.
        - validator: Function to select valid combinations of hyperparameters.
        """
        if "widths" not in hyperparameters:
            raise ValueError("Widths must be specified in hyperparameters.")

        if "optimizer" not in hyperparameters:
            raise ValueError("Optimizer must be specified in hyperparameters.")

        super().__init__(
            model_class=nn.MLP,
            hyperparameters=hyperparameters,
            validator=validator,
            process_count=process_count,
            random_seed=random_seed
        )
