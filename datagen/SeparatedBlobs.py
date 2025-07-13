import numpy as np

from datagen.MatrixSampler import MatrixSampler
from datagen.Postprocessors import *

import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class SepBlobConf:
    cx: float = 0
    cy: float = 0
    radius: float = 0.2


class SeparatedBlobs(MatrixSampler):

    def __init__(self, blobsConf=None, include_background=False, random_seed=None):
        if blobsConf is None:
            blobsConf = [SepBlobConf(-0.3, -0.3, 2), SepBlobConf(0.3, 0.3, 1)]

        self.blobConfs = blobsConf

        labels = {i+1: f'Blob {i+1}' for i in range(len(blobsConf))}
        if include_background:
            labels[0] = 'Background'

        super().__init__(labels=labels, random_seed=random_seed)

    def get_label(self, x, y):
        for i, bc in enumerate(self.blobConfs):
            if (x - bc.cx)**2 + (y - bc.cy)**2 <= bc.radius**2:
                return i + 1
        return 0

    def get_coordinate_transform(self, width, height):
        return CoordinateMapper(
            x_in_range=(-1, 1),
            y_in_range=(-1, 1),
            x_out_range=(0, width),
            y_out_range=(0, height)
        )

    def generate_point(self, width, height):
        bc = self.generator.choice(self.blobConfs)

        transform = self.get_coordinate_transform(width, height)

        cx, cy = transform((bc.cx, bc.cy))
        max_radius = bc.radius * min(width, height) / 2

        angle = self.generator.uniform(0, 2 * np.pi)
        radius = self.generator.uniform(0, max_radius)

        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)

        return np.array([x, y])

    def sample(self, num_samples=100):
        coords = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        image = np.vectorize(self.get_label)(coords[0], coords[1])

        self.point_generator = self.generate_point

        samples = super().sample(image, num_samples=num_samples)

        return samples

class RandomSeparatedBlobs(SeparatedBlobs):

    def __init__(self, blob_count=2, include_background=False, random_seed=None):
        blobs = []
        for _ in range(blob_count):
            cx = np.random.uniform(-1, 1)
            cy = np.random.uniform(-1, 1)

            min_distance = 2
            for other_bc in blobs:
                min_distance = min(min_distance, np.sqrt((cx - other_bc.cx)**2 + (cy - other_bc.cy)**2))
            radius = np.random.uniform(0, min_distance)

            blobs.append(SepBlobConf(cx, cy, radius))

        super().__init__(blobsConf=blobs, include_background=include_background, random_seed=random_seed)
