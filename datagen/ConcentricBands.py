import numpy as np

from datagen.SampleGenerator import SampleGenerator
from dataclasses import dataclass

@dataclass
class CBandConf:
    """Configuration describing a band around the origin."""
    
    radius: float = 0.5
    width: float = 0.1


class ConcentricBands(SampleGenerator):
    """
    Samples a set of concentric bands around the origin.
    """

    def __init__(self, bandsConf=None):
        """
        Initializes the concentric bands sampler.
        
        Parameters:
        - bandsConf: List of CBandConf objects defining the bands.
        """
        
        if bandsConf is None:
            bandsConf = [CBandConf(0.3, 0.1), CBandConf(0.6, 0.1)]

        self.bandsConf = bandsConf

        labels = {i: f'Band {i}' for i in range(len(bandsConf))}

        super().__init__(labels=labels)

    def _get_samples(self, num_samples):
        """
        Returns samples from the concentric bands.
        
        Parameters:
        - num_samples: Number of samples to generate.
        
        Yields:
        - (x, y, label) tuples where x and y are coordinates and label is the band label.
        """
        for _ in range(num_samples):
            band = np.random.choice(self.bandsConf)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Ensure the radius is uniformly distributed within the band
            radius = np.sqrt(np.random.uniform((band.radius-band.width / 2)**2, (band.radius+band.width / 2)**2))
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            label = self.labels[self.bandsConf.index(band)]
            yield x, y, label

class RandomConcentricBands(ConcentricBands):
    """
    Generates random concentric bands with specified parameters.
    """

    def __init__(self, num_bands=2, min_distance=0.1, variation=0.1):
        """
        Initializes random concentric bands.
        
        Parameters:
        - num_bands: Number of concentric bands.
        - min_distance: Minimum distance between the bands.
        - variation: How wide the bands can be, relative to the distance between them.
        """
        if (num_bands + 1) * min_distance > 1:
            raise ValueError("Too many bands for the given minimum distance.")

        # First we generate the radii of the bands
        radii = []
        min_radius = min_distance
        max_radius = 1 - num_bands * min_distance
        for i in range(num_bands):
            # Ensure the radius has enough distance from the previous band but also
            # leaves enough space for the remaining bands
            radius = np.random.uniform(min_radius, max_radius)
            radii.append(round(radius, 2))

            min_radius = radius + min_distance
            max_radius = 1 - (num_bands - i - 1) * min_distance

        # Now we calculate the widths of the bands based on the distances between them
        widths = []
        for i in range(num_bands):
            distance = min(radii[i] - (radii[i-1] if i > 0 else 0),
                          (radii[i+1] if i < num_bands - 1 else 1) - radii[i])
            distance = distance if distance < 1 else 0.5
            widths.append(round(distance * variation / 2, 2))

        # Create band configurations
        bandsConf = [CBandConf(radii[i], widths[i]) for i in range(num_bands)]
        
        super().__init__(bandsConf=bandsConf)
