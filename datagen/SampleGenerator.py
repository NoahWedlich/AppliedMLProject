import numpy as np
import pandas as pd

from datagen.Sampler import Sampler

class SampleGenerator(Sampler):
    
    def __init__(self, labels):
        super().__init__()
        
        self.labels = labels or {}
        
    def _get_samples(self, num_samples):
        raise NotImplementedError("Subclasses should implement this method.")
        
    def sample(self, num_samples=100):
        return pd.DataFrame(self._get_samples(num_samples), columns=['x', 'y', 'label'])