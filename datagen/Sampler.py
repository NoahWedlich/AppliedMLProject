from datagen.Preprocessors import PreProcessingChain
from datagen.Postprocessors import PostProcessingChain, CoordinateMapper


class Sampler:
    """
    The base class for all samplers. Provides methods to set and apply processors.
    """

    def __init__(self):
        """
        Initializes the Sampler with no preprocessor or postprocessor.
        """
        self.preprocessor = None
        self.postprocessor = None

    def set_preprocessor(self, preprocessor):
        """
        Sets the preprocessor for the sampler.

        Parameters:
        - preprocessor: An instance of a Preprocessor to be applied before sampling.
        """
        self.preprocessor = preprocessor

    def add_preprocessor(self, preprocessor):
        """
        Adds a preprocessor to the sampler to be applied before the existing ones.

        Parameters:
        - preprocessor: An instance of a Preprocessor to be added.
        """
        if preprocessor is not None:
            if self.preprocessor is None:
                self.preprocessor = preprocessor
            else:
                self.preprocessor = PreProcessingChain(
                    *(
                        preprocessor.get_preprocessors()[::-1]
                        + self.preprocessor.get_preprocessors()[::-1]
                    )
                )

    def apply_preprocessor(self, image):
        """
        Applies the preprocessor to the given image.

        Parameters:
        - image: The input image to be preprocessed.

        Returns:
        - The preprocessed image.
        """
        if self.preprocessor is None:
            return image
        return self.preprocessor(image)

    def set_postprocessor(self, postprocessor):
        """
        Sets the postprocessor for the sampler.

        Parameters:
        - postprocessor: An instance of a Postprocessor to be applied after sampling.
        """
        self.postprocessor = postprocessor

    def add_postprocessor(self, postprocessor):
        """
        Adds a postprocessor to the sampler to be applied after the existing ones.

        Parameters:
        - postprocessor: An instance of a Postprocessor to be added.
        """
        if postprocessor is not None:
            if self.postprocessor is None:
                self.postprocessor = postprocessor
            else:
                self.postprocessor = PostProcessingChain(
                    *(
                        postprocessor.get_postprocessors()[::-1]
                        + self.postprocessor.get_postprocessors()[::-1]
                    )
                )

    def apply_postprocessor(self, point, label):
        """
        Applies the postprocessor to the given point and label.

        Parameters:
        - point: The sample point to be processed (x, y).
        - label: The label associated with the sample point.

        Returns:
        - tuple: A tuple containing the processed point and label.
        """
        if self.postprocessor is None:
            return point, label
        return self.postprocessor(point, label)

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
            y_out_range=(-1, 1),
        )
