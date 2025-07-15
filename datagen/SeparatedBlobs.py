import numpy as np

from datagen.SampleGenerator import SampleGenerator

from dataclasses import dataclass


@dataclass
class SepBlobConf:
    """Configuration for a gaussian blob."""

    cx: float = 0
    cy: float = 0
    stddev: float = 1


class SeparatedBlobs(SampleGenerator):
    """
    Samples a set of separated blobs in the unit square.
    """

    def __init__(self, blobsConf=None):
        """
        Initializes the separated blobs sampler.

        Parameters:
        - blobsConf: List of SepBlobConf objects defining the blobs.
        """

        if blobsConf is None:
            blobsConf = [SepBlobConf(-0.3, -0.3, 2), SepBlobConf(0.3, 0.3, 1)]

        self.blobConfs = blobsConf

        labels = {i + 1: f"Blob {i}" for i in range(len(blobsConf))}

        super().__init__(labels=labels)

    def _get_samples(self, num_samples):
        """
        Returns samples from the separated blobs.

        Parameters:
        - num_samples: Number of samples to generate.

        Yields:
        - (x, y, label) tuples where x and y are coordinates and label is the blob label.
        """
        generated_samples = 0
        while generated_samples < num_samples:
            blob = np.random.choice(self.blobConfs)

            x = np.random.normal(blob.cx, blob.stddev)
            y = np.random.normal(blob.cy, blob.stddev)

            label = self.labels[self.blobConfs.index(blob) + 1]
            if -1 < x < 1 and -1 < y < 1:
                generated_samples += 1
                yield x, y, label


class RandomSeparatedBlobs(SeparatedBlobs):
    """
    Generates random separated blobs with specified parameters.
    """

    def __init__(self, blob_count=2):
        """
        Initializes random separated blobs.

        Parameters:
        - blob_count: Number of blobs to generate.
        """
        xs = np.random.uniform(-1, 1, blob_count)
        ys = np.random.uniform(-1, 1, blob_count)
        stddevs = np.random.uniform(0.1, 1, blob_count)

        blobConfs = [SepBlobConf(xs[i], ys[i], stddevs[i]) for i in range(blob_count)]

        super().__init__(blobsConf=blobConfs)
