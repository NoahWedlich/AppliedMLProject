import numpy as np
import pandas as pd

from datagen.Sampler import Sampler


class SampleGenerator(Sampler):
    """
    Base class for sampling a generated object.
    """

    def __init__(self, labels):
        """
        Initializes the sample generator with a dictionary of labels.

        Parameters:
        - labels: dictionary mapping label names to indices.
        """
        super().__init__()

        self.labels = labels or {}
        self.labels_inv = {v: k for k, v in self.labels.items()}

    def _get_samples(self, num_samples):
        raise NotImplementedError("Subclasses should implement this method.")

    def _get_samples_and_apply_postprocessors(self, num_samples):
        """
        Generates samples and then applies postprocessors to each sample.

        Parameters:
        - num_samples: number of samples to generate.
        """
        for sample in self._get_samples(num_samples):
            (x, y), label = self.apply_postprocesser(sample[:2], sample[2])

            index = self.labels_inv.get(label, -1)

            yield x, y, index, label

    def sample(self, num_samples=100):
        """
        Generates a DataFrame of samples with their coordinates and labels.

        Parameters:
        - num_samples: number of samples to generate.

        Returns:
        - pd.DataFrame: DataFrame containing the sample coordinates, labels, and display labels.
        """
        return pd.DataFrame(
            self._get_samples_and_apply_postprocessors(num_samples),
            columns=["x", "y", "label", "display_label"],
        )
    
    def sample_using_seed(self, num_samples=100, seed=None):
        """
        Generates samples using a specific seed for reproducibility.

        Parameters:
        - num_samples: number of samples to generate.
        - seed: random seed for reproducibility.

        Returns:
        - pd.DataFrame: DataFrame containing the sample coordinates, labels, and display labels.
        """
        if seed is not None:
            np.random.seed(seed)
        
        return self.sample(num_samples)