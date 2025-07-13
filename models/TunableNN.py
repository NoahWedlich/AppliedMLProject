
from AppliedML.courselib.models import nn

from TunableModel import TunableModel

class TunableNN(TunableModel):
    
    def __init__(self, hyperparameters, validator):
        if 'widths' not in hyperparameters:
            raise ValueError("Widths must be specified in hyperparameters.")
            
        if 'optimizer' not in hyperparameters:
            raise ValueError("Optimizer must be specified in hyperparameters.")
            
        super().__init__(
            model_class=nn.MLP,
            hyperparameters=hyperparameters,
            validator=validator,
        )