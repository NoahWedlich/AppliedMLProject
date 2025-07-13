import numpy as np

from datagen.MatrixSampler import MatrixSampler

class SampleGenerator(MatrixSampler):
    
    def __init__(self, labels, random_seed=None):
        super().__init__(labels=labels, random_seed=random_seed)
        
    def get_label(self, x, y):
        raise NotImplementedError("Subclasses should implement this method.")
        
    def sample(self, num_samples=100):
        coords = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        image = np.vectorize(self.get_label)(coords[0], coords[1])
        
        return super().sample(image, num_samples=num_samples)