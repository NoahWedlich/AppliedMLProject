import numpy as np

class Preprocessor:
    """
    Base class for preprocessors which modify input images.
    """
    
    def __call__(self, image):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_preprocessers(self):
        """
        Returns the chain of preprocessors associated with this instance.
        
        Returns:
        - list: A list containing this preprocessor instance.
        """
        return [self]

class PreProcessingChain(Preprocessor):
    """
    A chain of preprocessors that applies multiple transformations to the data.
    """
    
    def __init__(self, *preprocessors):
        """
        Initializes the chain with a list of preprocessors.
        
        Parameters:
        - preprocessors: variable number of preprocessors to be applied in order.
        """
        self.preprocessors = preprocessors
        
    def __call__(self, image):
        """
        Applies each preprocessor in the chain to the given image.
        
        Parameters:
        - image: the input image to be processed.
        
        Returns:
        - image: The processed image after applying all preprocessors.
        """
        for preprocessor in self.preprocessors:
            image = preprocessor(image)
        return image
        
    def get_preprocessers(self):
        """
        Returns the list of preprocessors in the chain.
        
        Returns:
        - list: A list of preprocessor instances in the order they will be applied.
        """
        return self.preprocessors

class Threshold(Preprocessor):
    """
    Applies a threshold to the input image, converting it to binary.
    """
    
    def __init__(self, threshold=0.5):
        """
        Initializes the threshold preprocessor.
        
        Parameters:
        - threshold: the threshold value to apply (default is 0.5).
        """
        self.threshold = threshold
        
    def __call__(self, image):
        """
        Applies the threshold to the input image.
        
        Parameters:
        - image: the input image to be thresholded.
        
        Returns:
        - image: The binary image after applying the threshold.
        """
        return np.vectorize(lambda c: 1 if c > self.threshold else 0)(image)
        
class ColorCollapse(Preprocessor):
    """
    Collapses the color channels of an image to a single channel using a mapping function.
    """
    
    def __init__(self, mapping):
        """
        Initializes the color collapse preprocessor with a mapping function.
        
        Parameters:
        - mapping: a function that maps the color channels to a single value.
        """
        self.mapping = mapping or (lambda x: x[0])
        
    def __call__(self, image):
        """
        Applies the color collapse mapping to the input image.
        """
        return np.apply_along_axis(self.mapping, 2, image)