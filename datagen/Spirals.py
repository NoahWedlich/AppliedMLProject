import numpy as np
from dataclasses import dataclass

from datagen.SampleGenerator import SampleGenerator
from datagen.Postprocessors import *

@dataclass
class SpiralConf:
    nBranches: int = 3
    width: float = 0.1
    innerRadius: float = 0.2
    outerRadius: float = 0.8
    rotations: float = 2
    
    def unpack(self):
        return (self.nBranches, self.width, self.innerRadius, self.outerRadius, self.rotations)

class Spirals(SampleGenerator):
    def __init__(self, spiralConfig=None):
        if spiralConfig is None:
            spiralConfig = SpiralConf()
            
        self.spiralConfig = spiralConfig

        labels = {i: f'Spiral {i}' for i in range(spiralConfig.nBranches)}

        super().__init__(labels=labels)
        
    def _get_samples(self, num_samples):
        sc = self.spiralConfig
        radius_delta = sc.outerRadius - sc.innerRadius
        
        two_pi = 2 * np.pi
        
        for _ in range(num_samples):
            branch = np.random.randint(0, sc.nBranches)
            
            start_angle = branch * (two_pi / sc.nBranches)
            rotated_angle = np.random.uniform(0, sc.rotations * two_pi)
            
            factor = rotated_angle / (sc.rotations * two_pi)
            
            radius_center = sc.innerRadius + factor * radius_delta
            radius_lb = radius_center - sc.width / 2
            radius_ub = radius_center + sc.width / 2
            
            radius = np.sqrt(np.random.uniform(radius_lb**2, radius_ub**2))
            
            angle = start_angle + rotated_angle
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            label = self.labels[branch]
            
            yield x, y, label