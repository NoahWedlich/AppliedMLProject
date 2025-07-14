
from AppliedML.courselib.models import svm

from TunableModel import TunableModel

class TunableKernelSVM(TunableModel):
    
    def __init__(self, hyperparameters):
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
            validator=self.validator
        )
        
    def validator(params):
        if params['kernel'] == 'rbf':
            return params['sigma'] > 0 and params['degree'] == 0 and params['intercept'] == 0
        elif params['kernel'] == 'polynomial':
            return params['sigma'] == 0 and params['degree'] > 0 and params['intercept'] >= 0
        else:
            return False
        
    def fit(self, X, y):
        return super().fit(X, y)