import numpy as np


class Postprocessor:
    """
    Base class for postprocessors which modify the sample points output by a sampler.
    """

    def __call__(self, point, label):
        raise NotImplementedError("Subclasses should implement this method.")

    def get_postprocessors(self):
        """
        Returns the chain of postprocessors associated with this instance.

        Returns:
        - list: A list containing this postprocessor instance.
        """
        return [self]


class PostProcessingChain(Postprocessor):
    """
    A chain of postprocessors that applies multiple transformations to the data.
    """

    def __init__(self, *postprocessors):
        """
        Initializes the chain with a list of postprocessors.

        Parameters:
        - postprocessors: variable number of postprocessors to be applied in order.
        """
        self.postprocessors = postprocessors

    def __call__(self, point, label):
        """
        Applies each postprocessor in the chain to the given point and label.

        Parameters:
        - point: the sample point to be processed (x, y).
        - label: the label associated with the sample point.

        Returns:
        - tuple: A tuple containing the processed point and label.
        """
        for postprocessor in self.postprocessors:
            point, label = postprocessor(point, label)

        return point, label

    def get_postprocessors(self):
        """
        Returns the list of postprocessors in the chain.

        Returns:
        - list: A list of postprocessor instances in the order they will be applied.
        """
        return self.postprocessors


class CoordinateMapper(Postprocessor):
    """
    Maps 2d coordinates from one space to another.
    """

    def __init__(self, x_in_range, y_in_range, x_out_range, y_out_range):
        """
        Initializes the coordinate mapper with input and output ranges.

        Parameters:
        - x_in_range: tuple (min, max) for input x coordinates.
        - y_in_range: tuple (min, max) for input y coordinates.
        - x_out_range: tuple (min, max) for output x coordinates.
        - y_out_range: tuple (min, max) for output y coordinates.
        """
        self.x_in_range = x_in_range
        self.y_in_range = y_in_range
        self.x_out_range = x_out_range
        self.y_out_range = y_out_range

    def __call__(self, point, label):
        """
        Maps the input point from the input coordinate space to the output coordinate space.

        Parameters:
        - point: the sample point to be mapped (x, y).
        - label: the label associated with the sample point.

        Returns:
        - tuple: A tuple containing the mapped point and the original label.
        """
        x_in, y_in = point

        ix_min, ix_max = self.x_in_range
        iy_min, iy_max = self.y_in_range

        ox_min, ox_max = self.x_out_range
        oy_min, oy_max = self.y_out_range

        x_out = ox_min + (x_in - ix_min) / (ix_max - ix_min) * (ox_max - ox_min)
        y_out = oy_min + (y_in - iy_min) / (iy_max - iy_min) * (oy_max - oy_min)

        return (x_out, y_out), label


class LabelSwitch(Postprocessor):
    """
    Randomly switches labels of points with a certain frequency.
    """

    def __init__(self, labels, noise_freq=0.5):
        """
        Initializes the label noise postprocessor.

        Parameters:
        - noise_freq: probability of applying noise to a point.
        - labels: list of possible labels to switch to.
        """
        self.noise_freq = noise_freq
        self.labels = labels

    def __call__(self, point, label):
        """
        Applies random noise to the sample point.

        Parameters:
        - point: the sample point to be modified (x, y).
        - label: the label associated with the sample point.

        Returns:
        - tuple: A tuple containing the original point and the new label.
        """
        new_label = label
        if np.random.rand() < self.noise_freq:
            new_label = np.random.choice(list(set(self.labels) - set([label])))
        return point, new_label


class LabelNoise(Postprocessor):
    """
    Randomly shifts the sample points with a certain frequency and noise level.
    """

    def __init__(self, noise_freq=0.5, noise_level=0.1):
        """
        Initializes the label noise postprocessor.

        Parameters:
        - noise_freq: probability of applying noise to a point.
        - noise_level: maximum amount of noise to add to each coordinate.
        """
        self.noise_freq = noise_freq
        self.noise_level = noise_level

    def __call__(self, point, label):
        """
        Applies random noise to the sample point.

        Parameters:
        - point: the sample point to be modified (x, y).
        - label: the label associated with the sample point.

        Returns:
        - tuple: A tuple containing the modified point and the original label.
        """
        new_x, new_y = point

        if np.random.rand() < self.noise_freq:
            new_x += np.random.uniform(-self.noise_level, self.noise_level)
            new_y += np.random.uniform(-self.noise_level, self.noise_level)

        return (new_x, new_y), label


class DomainShift(Postprocessor):
    """
    Applies a constant translation to the sample points.
    """

    def __init__(self, shift_x=0, shift_y=0):
        """
        Initializes the domain shift postprocessor.

        Parameters:
        - shift_x: amount to shift the x coordinate.
        - shift_y: amount to shift the y coordinate.
        """
        self.shift_x = shift_x
        self.shift_y = shift_y

    def __call__(self, point, label):
        """
        Applies a constant translation to the sample point.

        Parameters:
        - point: the sample point to be shifted (x, y).
        - label: the label associated with the sample point.

        Returns:
        - tuple: A tuple containing the shifted point and the original label.
        """
        new_x, new_y = point
        new_x += self.shift_x
        new_y += self.shift_y
        return (new_x, new_y), label
