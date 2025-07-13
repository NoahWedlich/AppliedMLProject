import numpy as np
import pandas as pd

from datagen.Sampler import Sampler

class SampleGenerator(Sampler):
    
    def __init__(self, labels):
        super().__init__()
        
        self.labels = labels or {}
        self.labels_inv = {v: k for k, v in self.labels.items()}
        
    def _get_samples(self, num_samples):
        raise NotImplementedError("Subclasses should implement this method.")
        
    def _get_samples_and_apply_postprocessors(self, num_samples):
        for sample in self._get_samples(num_samples):
            (x, y), label = self.apply_postprocesser(sample[:2], sample[2])
            
            index = self.labels_inv.get(label, -1)
            
            yield x, y, index, label
        
    def sample(self, num_samples=100):
        return pd.DataFrame(self._get_samples_and_apply_postprocessors(num_samples), columns=['x', 'y', 'label', 'disply-label'])