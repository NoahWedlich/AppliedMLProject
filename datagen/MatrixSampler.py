import numpy as np
import pandas as pd

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
                yield *self.postprocesser(point), self.labels[value]
            else:
                missed_samples += 1
        
    def sample(self, image, num_samples=100):
        image = self.preprocesser(image)
            
        return pd.DataFrame(self.get_samples(image, num_samples), columns=['x', 'y', 'label'])