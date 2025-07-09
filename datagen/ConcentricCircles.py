import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import random

class ConcentricCircles:
    def __init__(self, num_circles=2, num_samples_per_circle=100, variation=0.1):
        super().__init__()
        
        self.num_circles = num_circles
        self.num_samples_per_circle = num_samples_per_circle
        self.variation = max(0, min(1, variation))  # Ensure variation is between 0 and 1
        
    def generate_circle(self, radius, width):
        angles = np.random.uniform(0, 2 * np.pi, self.num_samples_per_circle)
        radii = radius + np.random.uniform(-width, width, self.num_samples_per_circle)
        
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        
        return x, y
        
    def generate_sample(self):
        step = 1
        
        min_radius = 0
        max_radius = 1
        
        radii = []
        
        for i in range(self.num_circles):
            radius = random.uniform(min_radius, max_radius)
            min_radius = radius + step
            max_radius = radius + 2*step
            radii.append(radius)
        radii = np.array(radii)
        
        differences = np.abs(np.diff(radii, prepend=0, append=0))
        min_difference = np.minimum(differences[:-1], differences[1:])
        widths = min_difference * self.variation / 2
        
        xs = np.array([])
        ys = np.array([])
        numeric_labels = np.array([], dtype=int)
        display_labels = np.array([], dtype=object)
        
        for i in range(self.num_circles):
            circleXs, circleYs = self.generate_circle(radii[i], widths[i])
            
            xs = np.append(xs, circleXs)
            ys = np.append(ys, circleYs)
            
            numeric_labels = np.append(numeric_labels, [i] * self.num_samples_per_circle)
            display_labels = np.append(display_labels, [f'Circle {i+1}'] * self.num_samples_per_circle)
            
        return pd.DataFrame({
            'x': xs,
            'y': ys,
            'numeric_label': numeric_labels,
            'display_label': display_labels
        })