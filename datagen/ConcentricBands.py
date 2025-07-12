import numpy as np

from datagen.MatrixSampler import MatrixSampler
from datagen.Postprocessors import *

class ConcentricBands(MatrixSampler):
    
    def __init__(self, bands=None, include_background=False, random_seed=None):
        if bands is None:
            bands = [(0.3, 0.1), (0.6, 0.1)]
            
        self.bands = bands
        
        labels = {i+1: f'Band {i+1}' for i in range(len(bands))}
        if include_background:
            labels[0] = 'Background'
        
        super().__init__(labels=labels, random_seed=random_seed)
        
    def get_label(self, x, y):
        for i, (radius, width) in enumerate(self.bands):
            if (radius - width)**2 <= x**2 + y**2 <= (radius + width)**2:
                return i + 1
        return 0
    
    def get_normalizer(self):
        return CoordinateMapper(
            x_in_range=(0, 100),
            y_in_range=(0, 100),
            x_out_range=(-1, 1),
            y_out_range=(-1, 1)
        )
        
    def sample(self, num_samples=100):
        coords = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        image = np.vectorize(self.get_label)(coords[0], coords[1])
        
        self.set_postprocesser(self.get_normalizer())
        
        return super().sample(image, num_samples=num_samples)
        
class RandomConcentricBands(ConcentricBands):
    
    def __init__(self, num_bands=2, min_distance=0.1, variation=0.1, include_background=False, random_seed=None):
        if (num_bands + 1) * min_distance > 1:
            raise ValueError("Too many bands for the given minimum distance.")
            
        temp_bands = [(0, 0) for _ in range(num_bands)]
        super().__init__(bands=temp_bands, include_background=include_background, random_seed=random_seed)
            
        radii = []
        min_radius = min_distance
        max_radius = 1 - num_bands * min_distance
        for i in range(num_bands):
            radius = self.generator.uniform(min_radius, max_radius)
            radii.append(round(radius, 2))
            
            min_radius = radius + min_distance
            max_radius = 1 - (num_bands - i - 1) * min_distance
            
        widths = []
        for i in range(num_bands):
            distance = min(radii[i] - (radii[i-1] if i > 0 else 0), 
                          (radii[i+1] if i < num_bands - 1 else 1) - radii[i])
            distance = distance if distance < 1 else 0.5
            widths.append(round(distance * variation / 2, 2))
            
        self.bands = [(radii[i], widths[i]) for i in range(num_bands)]