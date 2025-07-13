import numpy as np

from datagen.SampleGenerator import SampleGenerator
from dataclasses import dataclass

@dataclass
class CBandConf:
    radius: float = 0.5
    width: float = 0.1


class ConcentricBands(SampleGenerator):

    def __init__(self, bandsConf=None):
        if bandsConf is None:
            bandsConf = [CBandConf(0.3, 0.1), CBandConf(0.6, 0.1)]

        self.bandsConf = bandsConf

        labels = {i: f'Band {i}' for i in range(len(bandsConf))}

        super().__init__(labels=labels)

    def _get_samples(self, num_samples):
        for _ in range(num_samples):
            band = np.random.choice(self.bandsConf)
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.sqrt(np.random.uniform((band.radius-band.width / 2)**2, (band.radius+band.width / 2)**2))
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            label = self.labels[self.bandsConf.index(band)]
            yield x, y, label

class RandomConcentricBands(ConcentricBands):

    def __init__(self, num_bands=2, min_distance=0.1, variation=0.1, include_background=False, random_seed=None):
        if (num_bands + 1) * min_distance > 1:
            raise ValueError("Too many bands for the given minimum distance.")

        temp_bands = [CBandConf(0, 0) for _ in range(num_bands)]
        super().__init__(bandsConf=temp_bands, include_background=include_background, random_seed=random_seed)

        radii = []
        min_radius = min_distance
        max_radius = 1 - num_bands * min_distance
        for i in range(num_bands):
            radius = self.generator.uniform(min_radius, max_radius)
            radii.append(round(radius, 2))

            min_radius = radius + min_distance
            max_radius = 1 - (num_bands - i - 1) * min_distance

        widths = []
        for i in range(num_bands):
            distance = min(radii[i] - (radii[i-1] if i > 0 else 0),
                          (radii[i+1] if i < num_bands - 1 else 1) - radii[i])
            distance = distance if distance < 1 else 0.5
            widths.append(round(distance * variation / 2, 2))

        self.bandsConf = [CBandConf(radii[i], widths[i]) for i in range(num_bands)]
