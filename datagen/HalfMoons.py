import numpy as np

from datagen.SampleGenerator import SampleGenerator
from dataclasses import dataclass

@dataclass
class HalfMoonConf:
    radius: float = 0.5
    width: float = 0.05
    angle_range: tuple = (0, np.pi)
    centre: tuple = (0, 0)

class HalfMoons(SampleGenerator):
    def __init__(self, moonConfig=None):
        if moonConfig is None:
            moonConfig = [HalfMoonConf(centre=(-0.2,-0.2)),
                          HalfMoonConf(angle_range=(3.14,6.29), centre=(0.2,0.2))]

        self.moonConfig = moonConfig

        labels = {i: f'Moon {i}' for i in range(len(moonConfig))}

        super().__init__(labels=labels)

    def _get_samples(self, num_samples):
        for _ in range(num_samples):
            moon = np.random.choice(self.moonConfig)
            angle = np.random.uniform(moon.angle_range[0], moon.angle_range[1])
            radius = np.sqrt(np.random.uniform((moon.radius-moon.width / 2)**2,
                                               (moon.radius+moon.width / 2)**2))

            x = radius * np.cos(angle) + moon.centre[0]
            y = radius * np.sin(angle) + moon.centre[1]

            label = self.labels[self.moonConfig.index(moon)]
            yield x, y, label
