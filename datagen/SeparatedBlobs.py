import numpy as np

from datagen.SampleGenerator import SampleGenerator
from datagen.Postprocessors import *

from dataclasses import dataclass

@dataclass
class SepBlobConf:
    cx: float = 0
    cy: float = 0
    stddev: float = 1


class SeparatedBlobs(SampleGenerator):

    def __init__(self, blobsConf=None):
        if blobsConf is None:
            blobsConf = [SepBlobConf(-0.3, -0.3, 2), SepBlobConf(0.3, 0.3, 1)]

        self.blobConfs = blobsConf

        labels = {i+1: f'Blob {i}' for i in range(len(blobsConf))}

        super().__init__(labels=labels)
        
    def _get_samples(self, num_samples):
        for _ in range(num_samples):
            blob = np.random.choice(self.blobConfs)
            
            x = np.random.normal(blob.cx, blob.stddev)
            y = np.random.normal(blob.cy, blob.stddev)
            
            label = self.labels[self.blobConfs.index(blob) + 1]
            yield x, y, label
        
class RandomSeparatedBlobs(SeparatedBlobs):

    def __init__(self, blob_count=2):
        blobs = []
        for _ in range(blob_count):
            cx = np.random.uniform(-1, 1)
            cy = np.random.uniform(-1, 1)
            stddev = np.random.uniform(0.1, 1)

            blobs.append(SepBlobConf(cx, cy, stddev))

        super().__init__(blobsConf=blobs)
