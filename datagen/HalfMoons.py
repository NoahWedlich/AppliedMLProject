import numpy as np

from datagen.MatrixSampler import MatrixSampler
from dataclasses import dataclass

@dataclass
class HalfMoonConf:
    radius: float = 0.5
    width: float = 0.05
    angle_range: tuple = (0, np.pi)
    centre: tuple = (0, 0)

class HalfMoons(MatrixSampler):
    def __init__(self, moonConfig=None, include_background=False, random_seed=None):
        if moonConfig is None:
            moonConfig = [HalfMoonConf(centre=(-0.2,-0.2)),
                          HalfMoonConf(angle_range=(3.14,6.29), centre=(0.2,0.2))]

        self.moonConfig = moonConfig

        labels = {i+1: f'Moon {i+1}' for i in range(len(moonConfig))}
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

        for i, mc in enumerate(self.moonConfig):
            offset = (x - mc.centre[0], y - mc.centre[1])
            if (mc.radius - mc.width)**2 <= offset[0]**2 + offset[1]**2 <= (mc.radius + mc.width)**2:
                angle = np.arctan2(offset[1], offset[0])
                if in_angle_range(angle, mc.angle_range):
                    return i + 1
        return 0

    def sample(self, num_samples=100):
        coords = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        image = np.vectorize(self.get_label)(coords[0], coords[1])

        return super().sample(image, num_samples=num_samples)
