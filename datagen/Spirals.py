import numpy as np

from MatrixSampler import MatrixSampler
from Postprocessors import *

class Spirals(MatrixSampler):

    def __init__(self, spirals=None, include_background=False, random_seed=None):
        if spirals is None:
            spirals = [3, 0.1, 0.2, 0.8, 2]

        self.spirals = spirals

        labels = {i+1: f'Spiral {i+1}' for i in range(len(spirals))}
        if include_background:
            labels[0] = 'Background'

        super().__init__(labels=labels, random_seed=random_seed)

    def get_label(self, x, y):
        [nBranches, width, innerRadius, outerRadius, rotations] = self.spirals
        partOfTurn = np.arctan2(y, x) / (2 * np.pi)
        spiralDist = (outerRadius - innerRadius) / rotations
        for i in range(nBranches):
            for j in range(rotations):
                if abs(np.sqrt(x**2 + y**2) - (innerRadius + spiralDist * (((partOfTurn + i/nBranches) % 1) + j))) <= width:
                    return i + 1
        return 0

    def get_normalizer(self):
        return CoordinateMapper(
            x_in_range=(0, 100),
            y_in_range=(0, 100),
            x_out_range=(-1, 1),
            y_out_range=(-1, 1)
        )

    def sample(self, num_samples=100):
        coords = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        image = np.vectorize(self.get_label)(coords[0], coords[1])

        self.set_postprocesser(self.get_normalizer())

        return super().sample(image, num_samples=num_samples)
