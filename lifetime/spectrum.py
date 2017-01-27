import numpy as np


def normalise(x, y, tail_lower=1410, tail_upper=1690):
    """Normalise x to maximum value after removing baseline noise from tail averages"""
    loc = np.where((x < tail_lower) | (x > tail_upper))
    y -= np.mean(y[loc])  # Subtract baseline from tails
    # Normalise
    y /= max(y)
    return x, y