import numpy as np


def normalise(x, y, tail_lower=1430, tail_upper=1670, intensity=True):
    """Normalise x to maximum value after removing baseline noise from tail averages"""
    loc = np.where((x < tail_lower) | (x > tail_upper))
    y -= np.mean(y[loc])  # Subtract baseline from tails
    # Normalise
    if intensity:
        y /= max(y)
    return x, y