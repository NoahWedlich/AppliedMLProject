import numpy as np

from MatrixSampler import MatrixSampler
from Postprocessors import *

class HalfMoons(MatrixSampler):

    def __init__(self, moons=None, include_background=False, random_seed=None):
        if moons is None:
            moons = [(0.5, 0.05, (0,3.14), (-0.2,-0.2)),(0.5, 0.05, (3.14,6.29), (0.2,0.2))]

        self.moons = moons

        labels = {i+1: f'Moon {i+1}' for i in range(len(moons))}
        if include_background:
            labels[0] = 'Background'

        super().__init__(labels=labels, random_seed=random_seed)

    def get_label(self, x, y):
        def in_angle_range(angle, angle_range):
            angle = angle % (2 * np.pi)
            angle_range = (angle_range[0] % (2 * np.pi), angle_range[1] % (2 * np.pi))
            if angle_range[0] <= angle_range[1]:
                return angle_range[0] <= angle <= angle_range[1]
            else:
                return angle >= angle_range[0] or angle <= angle_range[1]

        for i, (radius, width, angle_range, centre) in enumerate(self.moons):
            offset = (x - centre[0], y - centre[1])
            if (radius - width)**2 <= offset[0]**2 + offset[1]**2 <= (radius + width)**2:
                angle = np.arctan2(offset[1], offset[0])
                if in_angle_range(angle, angle_range):
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
