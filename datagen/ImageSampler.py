import numpy as np
import pandas as pd
from PIL import Image

from datagen.Sampler import Sampler
from datagen.Preprocessors import ColorCollapse, Threshold


class ImageSampler(Sampler):
    """
    Randomly samples points from an image based on pixel values.
    """

    @staticmethod
    def open_image(image_path):
        """
        Opens an image file and converts it to a numpy array.

        Parameters:
        - image_path: str, path to the image file.

        Returns:
        - np.ndarray: Image data as a numpy array.
        """
        image = Image.open(image_path)
        return np.array(image)

    def __init__(self, pallete, labels, image):
        """
        Initializes the ImageSampler with a color pallete, labels, and an image.

        Parameters:
        - pallete: dict, mapping of color tuples to classes.
        - labels: dict, mapping classes to labels.
        - image: np.ndarray, the image data to sample from.
        """
        super().__init__()

        self.image = image
        self.width, self.height = image.shape[1], image.shape[0]

        self.pallete = pallete or {}
        self.labels = labels or {}

        self.set_preprocessor(
            ColorCollapse(lambda c: self.pallete.get(tuple(c.tolist()), -1))
        )
        self.set_postprocessor(self.get_normalizer(0, self.width, 0, self.height))

    def apply_bw_filter(self, threshold=0.5):
        """
        Applies a black-and-white filter to the image based on a threshold.

        Parameters:
        - threshold: float, threshold for converting to black-and-white.
        """
        self.add_preprocessor(Threshold(threshold))

    def _get_samples(self, image, num_samples):
        """
        Generates random samples from the image.

        Parameters:
        - image: np.ndarray, the image data to sample from.
        - num_samples: int, number of samples to generate.

        Yields:
        - tuple: A tuple containing the x, y coordinates, the class and the display label.
        """
        generated_samples = 0

        while generated_samples < num_samples:
            point = np.random.random((2,)) * np.array([self.width, self.height])
            pixel = point.astype(int)

            if (
                pixel[0] < 0
                or pixel[0] >= self.width
                or pixel[1] < 0
                or pixel[1] >= self.height
            ):
                continue

            value = image[pixel[1], pixel[0]]
            if value in self.labels:
                generated_samples += 1
                normalized_point = self.apply_preprocessor(point)
                yield *normalized_point, value, self.labels[value]

    def sample(self, num_samples=100):
        """
        Samples points from the image and returns them as a DataFrame.

        Parameters:
        - num_samples: int, number of samples to generate.

        Returns:
        - pd.DataFrame: DataFrame containing the sampled points, their classes, and labels.
        """
        image = self.apply_preprocessor(self.image)
        return pd.DataFrame(
            self._get_samples(image, num_samples), columns=["x", "y", "class", "label"]
        )
