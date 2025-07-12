
class PostProcessingChain:
    def __init__(self, *postprocessors):
        self.postprocessors = postprocessors

    def __call__(self, point):
        for postprocessor in self.postprocessors:
            point = postprocessor(point)
        return point

class CoordinateMapper:
    
    def __init__(self, x_in_range, y_in_range, x_out_range, y_out_range):
        self.x_in_range = x_in_range
        self.y_in_range = y_in_range
        self.x_out_range = x_out_range
        self.y_out_range = y_out_range
        
    def __call__(self, point):
        x_in, y_in = point
        
        ix_min, ix_max = self.x_in_range
        iy_min, iy_max = self.y_in_range
        
        ox_min, ox_max = self.x_out_range
        oy_min, oy_max = self.y_out_range
        
        x_out = ox_min + (x_in - ix_min) / (ix_max - ix_min) * (ox_max - ox_min)
        y_out = oy_min + (y_in - iy_min) / (iy_max - iy_min) * (oy_max - oy_min)
        
        return x_out, y_out