import numpy as np
from dataclasses import dataclass

from datagen.SampleGenerator import SampleGenerator
from datagen.Postprocessors import *


@dataclass
class SpiralConf:
    """Configuration describing a spiral."""

    nBranches: int = 3
    width: float = 0.1
    innerRadius: float = 0.2
    outerRadius: float = 0.8
    rotations: float = 2


class Spirals(SampleGenerator):
    """
    Samples a sprial pattern with multiple branches.
    """

    def __init__(self, spiralConfig=None):
        """
        Initializes the spiral sampler.

        Parameters:
        - spiralConfig: SpiralConf object defining the spiral parameters.
        """
        if spiralConfig is None:
            spiralConfig = SpiralConf()

        self.spiralConfig = spiralConfig

        labels = {i: f"Spiral {i}" for i in range(spiralConfig.nBranches)}

        super().__init__(labels=labels)

    def _get_samples(self, num_samples):
        """
        Returns samples from the spiral pattern.

        Parameters:
        - num_samples: Number of samples to generate.

        Yields:
        - (x, y, label) tuples where x and y are coordinates and label is the spiral branch label.
        """
        sc = self.spiralConfig
        radius_delta = sc.outerRadius - sc.innerRadius

        two_pi = 2 * np.pi

        for _ in range(num_samples):
            # Select a random branch
            branch = np.random.randint(0, sc.nBranches)

            # Choose some rotation and find the offset of the current branch
            start_angle = branch * (two_pi / sc.nBranches)
            rotated_angle = np.random.uniform(0, sc.rotations * two_pi)

            factor = rotated_angle / (sc.rotations * two_pi)

            # Calculate the range the radius can take
            radius_center = sc.innerRadius + factor * radius_delta
            radius_lb = radius_center - sc.width / 2
            radius_ub = radius_center + sc.width / 2

            # Ensure the radius is uniformly distributed within the band
            radius = np.sqrt(np.random.uniform(radius_lb**2, radius_ub**2))

            # Calculate the angle for the current sample
            angle = start_angle + rotated_angle

            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            label = self.labels[branch]

            yield x, y, label
