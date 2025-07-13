import numpy as np

from SampleGenerator import SampleGenerator
from Postprocessors import *

class Spirals(SampleGenerator):

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
            pointRad = np.sqrt(x**2 + y**2)
            pointRadOff = pointRad - spiralDist * ((partOfTurn + i/nBranches) % 1)
            if innerRadius <= pointRad <= outerRadius and (pointRadOff % spiralDist) <= width:
                return i + 1
        return 0
