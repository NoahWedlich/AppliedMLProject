import numpy as np
from dataclasses import dataclass

from SampleGenerator import SampleGenerator

@dataclass
class SpiralConf:
    nBranches: int = 3
    width: float = 0.1
    innerRadius: float = 0.2
    outerRadius: float = 0.8
    rotations: float = 2

class Spirals(SampleGenerator):
    def __init__(self, spiralConfig, include_background=False, random_seed=None):
        self.spiralConfig = spiralConfig

        labels = {i+1: f'Spiral {i+1}' for i in range(spiralConfig.nBranches)}
        if include_background:
            labels[0] = 'Background'

        super().__init__(labels=labels, random_seed=random_seed)

    def get_label(self, x, y):
        sc = self.spiralConfig
        partOfTurn = np.arctan2(y, x) / (2 * np.pi)
        spiralDist = (sc.outerRadius - sc.innerRadius) / sc.rotations
        for i in range(sc.nBranches):
            pointRad = np.sqrt(x**2 + y**2)
            pointRadOff = pointRad - spiralDist * ((partOfTurn + i/sc.nBranches) % 1)
            if sc.innerRadius <= pointRad <= sc.outerRadius and (pointRadOff % spiralDist) <= sc.width:
                return i + 1
        return 0
