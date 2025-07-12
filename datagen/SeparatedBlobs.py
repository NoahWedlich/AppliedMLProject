import numpy as np

from datagen.MatrixSampler import MatrixSampler
from datagen.Postprocessors import *

import matplotlib.pyplot as plt

class SeparatedBlobs(MatrixSampler):
    
    def __init__(self, blobs=None, include_background=False, random_seed=None):
        if blobs is None:
            blobs = [(-0.3, -0.3, 2), (0.3, 0.3, 1)]
            
        self.blobs = blobs
        
        labels = {i+1: f'Blob {i+1}' for i in range(len(blobs))}
        if include_background:
            labels[0] = 'Background'
            
        super().__init__(labels=labels, random_seed=random_seed)
        
    def get_label(self, x, y):
        for i, (cx, cy, radius) in enumerate(self.blobs):
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                return i + 1
        return 0
        
    def get_coordinate_transform(self, width, height):
        return CoordinateMapper(
            x_in_range=(-1, 1),
            y_in_range=(-1, 1),
            x_out_range=(0, width),
            y_out_range=(0, height)
        )
        
    def generate_point(self, width, height):
        blob = self.generator.choice(self.blobs)
        
        transform = self.get_coordinate_transform(width, height)
        
        cx, cy, max_radius = blob
        cx, cy = transform((cx, cy))
        max_radius = max_radius * min(width, height) / 2
        
        angle = self.generator.uniform(0, 2 * np.pi)
        radius = self.generator.uniform(0, max_radius)
        
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        
        return np.array([x, y])
        
    def sample(self, num_samples=100):
        coords = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        image = np.vectorize(self.get_label)(coords[0], coords[1])
        
        self.point_generator = self.generate_point
        
        samples = super().sample(image, num_samples=num_samples)
        
        return samples
        
class RandomSeparatedBlobs(SeparatedBlobs):
    
    def __init__(self, blob_count=2, include_background=False, random_seed=None):
        blobs = []
        for _ in range(blob_count):
            cx = np.random.uniform(-1, 1)
            cy = np.random.uniform(-1, 1)
            
            min_distance = 2
            for (other_cx, other_cy, _) in blobs:
                min_distance = min(min_distance, np.sqrt((cx - other_cx)**2 + (cy - other_cy)**2))
            radius = np.random.uniform(0, min_distance)
            
            blobs.append((cx, cy, radius))
            
        super().__init__(blobs=blobs, include_background=include_background, random_seed=random_seed)
        
sampler = SeparatedBlobs(blobs=[(-0.8, 0, 0.2), (0.8, 0, 0.2)], include_background=False)

plt.figure(figsize=(8, 8))
samples = sampler.sample(num_samples=1000)
plt.scatter(samples['x'], samples['y'], c=samples['label'].astype('category').cat.codes, cmap='viridis', edgecolor='k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.title('Separated Blobs Sample')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()