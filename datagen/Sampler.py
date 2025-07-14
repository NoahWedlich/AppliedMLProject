
from datagen.Preprocessors import *
from datagen.Postprocessors import *

class Sampler:
    """
    The base class for all samplers. Provides methods to set and apply processors.
    """
    
    def __init__(self):
        """
        Initializes the Sampler with no preprocesser or postprocesser.
        """
        self.preprocesser = None
        self.postprocesser = None
        
    def set_preprocesser(self, preprocesser):
        """
        Sets the preprocesser for the sampler.
        
        Parameters:
        - preprocesser: An instance of a Preprocessor to be applied before sampling.
        """
        self.preprocesser = preprocesser
            
    def add_preprocesser(self, preprocesser):
        """
        Adds a preprocesser to the sampler to be applied before the existing ones.
        
        Parameters:
        - preprocesser: An instance of a Preprocessor to be added.
        """
        if preprocesser is not None:
            if self.preprocesser is None:
                self.preprocesser = preprocesser
            else:
                self.preprocesser = PreProcessingChain(
                    *(preprocesser.get_preprocessers()[::-1] +
                    self.preprocesser.get_preprocessers()[::-1])
                )
                
    def apply_preprocesser(self, image):
        """
        Applies the preprocesser to the given image.
        
        Parameters:
        - image: The input image to be preprocessed.
        
        Returns:
        - The preprocessed image.
        """
        if self.preprocesser is None:
            return image
        return self.preprocesser(image)
        
    def set_postprocesser(self, postprocesser):
        """
        Sets the postprocesser for the sampler.
        
        Parameters:
        - postprocesser: An instance of a Postprocessor to be applied after sampling.
        """
        self.postprocesser = postprocesser
            
    def add_postprocesser(self, postprocesser):
        """
        Adds a postprocesser to the sampler to be applied after the existing ones.
        
        Parameters:
        - postprocesser: An instance of a Postprocessor to be added.
        """
        if postprocesser is not None:
            if self.postprocesser is None:
                self.postprocesser = postprocesser
            else:
                self.postprocesser = PostProcessingChain(
                    *(postprocesser.get_postprocessors()[::-1] +
                    self.postprocesser.get_postprocessors()[::-1])
                )
                
    def apply_postprocesser(self, point, label):
        """
        Applies the postprocesser to the given point and label.
        
        Parameters:
        - point: The sample point to be processed (x, y).
        - label: The label associated with the sample point.
        
        Returns:
        - tuple: A tuple containing the processed point and label.
        """
        if self.postprocesser is None:
            return point, label
        return self.postprocesser(point, label)
        
    def get_normalizer(self, x_range, y_range):
        """
        Returns a CoordinateMapper that normalizes coordinates to the range (-1, 1).
        
        Parameters:
        - x_range: tuple (min, max) for input x coordinates.
        - y_range: tuple (min, max) for input y coordinates.
        
        Returns:
        - CoordinateMapper: An instance that maps coordinates from the input range to (-1, 1).
        """
        return CoordinateMapper(
            x_in_range=x_range,
            y_in_range=y_range,
            x_out_range=(-1, 1),
            y_out_range=(-1, 1)
        )