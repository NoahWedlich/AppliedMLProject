
from datagen.Preprocessors import *
from datagen.Postprocessors import *

class Sampler:
    
    def __init__(self):
        self.preprocesser = None
        self.postprocesser = None
        
    def set_preprocesser(self, preprocesser):
        self.preprocesser = preprocesser
            
    def add_preprocesser(self, preprocesser):
        if preprocesser is not None:
            if self.preprocesser is None:
                self.preprocesser = preprocesser
            else:
                self.preprocesser = PreProcessingChain(
                    *(preprocesser.get_preprocessers()[::-1] +
                    self.preprocesser.get_preprocessers()[::-1])
                )
                
    def apply_preprocesser(self, image):
        if self.preprocesser is None:
            return image
        return self.preprocesser(image)
        
    def set_postprocesser(self, postprocesser):
        self.postprocesser = postprocesser
            
    def add_postprocesser(self, postprocesser):
        if postprocesser is not None:
            if self.postprocesser is None:
                self.postprocesser = postprocesser
            else:
                self.postprocesser = PostProcessingChain(
                    *(postprocesser.get_postprocessors()[::-1] +
                    self.postprocesser.get_postprocessors()[::-1])
                )
                
    def apply_postprocesser(self, point, label):
        if self.postprocesser is None:
            return point, label
        return self.postprocesser(point, label)
        
    def get_normalizer(self, x_range, y_range):
        return CoordinateMapper(
            x_in_range=x_range,
            y_in_range=y_range,
            x_out_range=(-1, 1),
            y_out_range=(-1, 1)
        )