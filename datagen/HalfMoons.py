import numpy as np

from datagen.SampleGenerator import SampleGenerator
from dataclasses import dataclass


@dataclass
class HalfMoonConf:
    """Configuration for a half-moon shape."""

    radius: float = 0.5
    width: float = 0.05
    angle_range: tuple = (0, np.pi)
    centre: tuple = (0, 0)


class HalfMoons(SampleGenerator):
    """
    Samples a set of half-moon shapes.
    """

    def __init__(self, moonConfig=None):
        """
        Initializes the half-moons sampler.

        Parameters:
        - moonConfig: List of HalfMoonConf objects defining the half-moons.
        """
        if moonConfig is None:
            moonConfig = [
                HalfMoonConf(centre=(-0.2, -0.2)),
                HalfMoonConf(angle_range=(3.14, 6.29), centre=(0.2, 0.2)),
            ]

        self.moonConfig = moonConfig

        labels = {i: f"Moon {i}" for i in range(len(moonConfig))}

        super().__init__(labels=labels)

    def _get_samples(self, num_samples):
        """
        Returns samples from the half-moons.

        Parameters:
        - num_samples: Number of samples to generate.

        Yields:
        - (x, y, label) tuples where x and y are coordinates and label is the half-moon label.
        """
        for _ in range(num_samples):
            moon = np.random.choice(self.moonConfig)
            angle = np.random.uniform(moon.angle_range[0], moon.angle_range[1])

            # Ensure the radius is uniformly distributed within the half-moon
            radius = np.sqrt(
                np.random.uniform(
                    (moon.radius - moon.width / 2) ** 2,
                    (moon.radius + moon.width / 2) ** 2,
                )
            )

            x = radius * np.cos(angle) + moon.centre[0]
            y = radius * np.sin(angle) + moon.centre[1]

            label = self.labels[self.moonConfig.index(moon)]
            yield x, y, label
