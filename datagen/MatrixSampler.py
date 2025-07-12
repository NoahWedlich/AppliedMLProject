import numpy as np
import pandas as pd

from datagen.Postprocessors import *

class MatrixSampler:
    
    def __init__(self, labels, random_seed=None):
        self.labels = labels
        
        self.preprocesser = lambda x: x
        self.postprocesser = lambda x: x
        
        self.random_seed = random_seed
        self.generator = np.random.default_rng(random_seed)
        self.point_generator = self.get_default_generator()
        
    def get_default_generator(self):
        return lambda w, h: self.generator.random((2,)) * np.array([w, h])
        
    def get_normalizer(self, width, height):
        return CoordinateMapper(
            x_in_range=(0, width),
            y_in_range=(0, height),
            x_out_range=(-1, 1),
            y_out_range=(-1, 1)
        )
        
    def set_preprocesser(self, preprocesser):
        if preprocesser is not None:
            self.preprocesser = preprocesser
        else:
            self.preprocesser = lambda x: x
            
    def set_postprocesser(self, postprocesser):
        if postprocesser is not None:
            self.postprocesser = postprocesser
        else:
            self.postprocesser = lambda x: x
                
    def get_samples(self, image, num_samples):
        generated_samples = 0
        missed_samples = 0
        
        while generated_samples < num_samples:
            point = self.point_generator(image.shape[1], image.shape[0])
            cords = point.astype(int)
            
            if cords[0] < 0 or cords[0] >= image.shape[0] or cords[1] < 0 or cords[1] >= image.shape[1]:
                missed_samples += 1
                continue
            
            value = image[cords[1], cords[0]]
            
            if value in self.labels.keys():
                generated_samples += 1
                normalized_point = self.get_normalizer(image.shape[1], image.shape[0])(point)
                yield *self.postprocesser(normalized_point), self.labels[value]
            else:
                missed_samples += 1
                
        print(f"Generated {generated_samples} samples, missed {missed_samples} samples.")
        
    def sample(self, image, num_samples=100):
        image = self.preprocesser(image)
            
        return pd.DataFrame(self.get_samples(image, num_samples), columns=['x', 'y', 'label'])