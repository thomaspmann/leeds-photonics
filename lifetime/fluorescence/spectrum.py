import numpy as np
from .lifetime import normalise


def remove_spectrum_noise(x, y, lb=1430, ub=1670):
    """
    Remove noise from spectrum measurement by averaging noise in tails.
    :param x: Wavelength array
    :param y: Intensity array
    :param lb: lower bound of tail below which to average
    :param ub: upper bound of tail above which to average
    :return: Noise free spectrum intensity, y
    """
    loc = np.where((x < lb) | (x > ub))
    y -= np.mean(y[loc])
    return y