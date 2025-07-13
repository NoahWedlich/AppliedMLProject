import numpy as np
import pandas as pd
from PIL import Image

from datagen.Sampler import Sampler
from datagen.Preprocessors import *
from datagen.Postprocessors import *

class ImageSampler(Sampler):
    @staticmethod
    def open_image(image_path):
        image = Image.open(image_path)
        return np.array(image)
    
    def __init__(self, pallete, labels, image):
        super().__init__()
        
        self.image = image
        self.width, self.height = image.shape[1], image.shape[0]
        
        self.pallete = pallete or {}
        self.labels = labels or {}
        
        self.set_preprocesser(ColorCollapse(lambda c: self.pallete.get(tuple(c.tolist()), -1)))
        self.set_postprocesser(self.get_normalizer(0, self.width, 0, self.height))
        
    def apply_bw_filter(self, threshold=0.5):
        self.add_preprocesser(Threshold(threshold))
        
    def _get_samples(self, image, num_samples):
        generated_samples = 0
        
        while generated_samples < num_samples:
            point = np.random.random((2,)) * np.array([self.width, self.height])
            pixel = point.astype(int)
            
            if pixel[0] < 0 or pixel[0] >= self.width or pixel[1] < 0 or pixel[1] >= self.height:
                continue
            
            value = image[pixel[1], pixel[0]]
            if value in self.labels:
                generated_samples += 1
                normalized_point = self.apply_preprocesser(point)
                yield *normalized_point, self.labels[value]
        
    def sample(self, num_samples=100):
        image = self.apply_preprocesser(self.image)
        return pd.DataFrame(self._get_samples(image, num_samples), columns=['x', 'y', 'label'])