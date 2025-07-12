import numpy as np
from PIL import Image

from datagen.MatrixSampler import MatrixSampler
from datagen.Postprocessors import *

class ImageSampler(MatrixSampler):
    @staticmethod
    def open_image(image_path):
        image = Image.open(image_path)
        return np.array(image)
    
    def __init__(self, pallete, labels, image, random_seed=None):
        self.image = np.apply_along_axis(lambda x: pallete.get(tuple(x.tolist()), -1), 2, image)
        self.image = np.flip(self.image, axis=0)
        
        super().__init__(labels=labels, random_seed=random_seed)
        
    def apply_bw_filter(self, threshold=0.5):
        self.set_preprocesser(lambda img:
            np.vectorize(lambda c: 1 if c > threshold else 0)(img)
        )
        
    def sample(self, num_samples=100):
        return super().sample(
            image=self.image,
            num_samples=num_samples
        )