import numpy as np

class Preprocessor:
    def __call__(self, image):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_preprocessers(self):
        return [self]

class PreProcessingChain(Preprocessor):
    def __init__(self, *preprocessors):
        self.preprocessors = preprocessors
        
    def __call__(self, image):
        for preprocessor in self.preprocessors:
            image = preprocessor(image)
        return image
        
    def get_preprocessers(self):
        return self.preprocessors

class Threshold(Preprocessor):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def __call__(self, image):
        return np.vectorize(lambda c: 1 if c > self.threshold else 0)(image)
        
class ColorCollapse(Preprocessor):
    def __init__(self, mapping):
        self.mapping = mapping or (lambda x: x[0])
        
    def __call__(self, image):
        return np.apply_along_axis(self.mapping, 2, image)