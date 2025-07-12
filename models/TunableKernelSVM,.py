
from AppliedML.courselib.models import svm

from TunableModel import TunableModel

class TunableKernelSVM(TunableModel):
    
    def __init__(self, hyperparameters, validator=None, random_seed=None):
        kernel = hyperparameters.get('kernel', None)
        if kernel is None:
            raise ValueError("Kernel must be specified in hyperparameters.")
            
        if kernel == 'rbf' and 'sigma' not in hyperparameters:
            raise ValueError("RBF kernel requires 'sigma' in hyperparameters.")
        if kernel == 'polynomial' and ('degree' not in hyperparameters or 'intercept' not in hyperparameters):
            raise ValueError("Polynomial kernel requires 'degree' and 'intercept' in hyperparameters.")
        
        super().__init__(
            model_class=svm.BinaryKernelSVM,
            hyperparameters=hyperparameters,
            validator=validator,
            random_seed=random_seed
        )
        
    def fit(self, X, y):
        return super().fit(X, y)