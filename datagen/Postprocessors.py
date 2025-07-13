import numpy as np

class Postprocessor:
    def __call__(self, point, label):
        raise NotImplementedError("Subclasses should implement this method.")
        
    def get_postprocessors(self):
        return [self]

class PostProcessingChain(Postprocessor):
    def __init__(self, *postprocessors):
        self.postprocessors = postprocessors

    def __call__(self, point, label):
        for postprocessor in self.postprocessors:
            point, label = postprocessor(point, label)
        return point, label
        
    def get_postprocessors(self):
        return self.postprocessors

class CoordinateMapper(Postprocessor):
    def __init__(self, x_in_range, y_in_range, x_out_range, y_out_range):
        self.x_in_range = x_in_range
        self.y_in_range = y_in_range
        self.x_out_range = x_out_range
        self.y_out_range = y_out_range
        
    def __call__(self, point, label):
        x_in, y_in = point
        
        ix_min, ix_max = self.x_in_range
        iy_min, iy_max = self.y_in_range
        
        ox_min, ox_max = self.x_out_range
        oy_min, oy_max = self.y_out_range
        
        x_out = ox_min + (x_in - ix_min) / (ix_max - ix_min) * (ox_max - ox_min)
        y_out = oy_min + (y_in - iy_min) / (iy_max - iy_min) * (oy_max - oy_min)
        
        return (x_out, y_out), label
        
class LabelNoise(Postprocessor):
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
        
    def __call__(self, point, label):
        new_x, new_y = point
        new_x += np.random.uniform(-self.noise_level, self.noise_level)
        new_y += np.random.uniform(-self.noise_level, self.noise_level)
        return (new_x, new_y), label
        
class DomainShift(Postprocessor):
    def __init__(self, shift_x=0, shift_y=0):
        self.shift_x = shift_x
        self.shift_y = shift_y
        
    def __call__(self, point, label):
        new_x, new_y = point
        new_x += self.shift_x
        new_y += self.shift_y
        return (new_x, new_y), label