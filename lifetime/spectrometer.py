import numpy as np
from scipy.optimize import curve_fit, minimize


def shift_time(x, length):
    """Shift time axis to the left. USed to account for pump"""
    # Shift time axis
    x -= length
    return x


def reject_time(x, y, reject_start=0, reject_end=0):
    # Reject Time
    ind = np.where((x >= reject_start) & (x < x[-1] - reject_end))
    x = x[ind]
    y = y[ind]
    return x, y


def normalise(y, point="max"):
    # Subtract baseline noise
    y -= min(y)
    # Normalise to start value
    if point == "max":
        y /= max(y)
    elif point == "start":
        y /= y[0]
    else:
        raise ValueError("point option must be either 'max' or 'start'.")
    return y


def remove_spectrum_noise(x, y, tail_lower=1430, tail_upper=1670):
    """Normalise x to maximum value after removing baseline noise from tail averages"""
    loc = np.where((x < tail_lower) | (x > tail_upper))
    y -= np.mean(y[loc])
    return x, y


def decay_fn(t, a, tau, c):
    """Mono-exponential fitting function. t is the time."""
    return a * np.exp(-t / tau) + c


def decay_fn2(t, a1, tau1, a2, tau2, c):
    """Duo-exponential fitting function. t is the time."""
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c


def fit_decay(x, y):
    """
    Function to fit the data, y with a mono-exponential decay using Levenberg-Marquardt algorithm.
    Return fitting parameters [a, tau, c].
    """

    # Guess initial fitting parameters
    t_loc = np.where(y <= y[0] / np.e)[0][0]
    tau = x[t_loc]
    p0 = [max(y), tau, min(y)]

    # Fitting
    try:
        # Fit using Levenberg-Marquardt algorithm
        popt, pcov = curve_fit(decay_fn, x, y, p0=p0)
        # Error in coefficients to 1 std.
        perr = np.sqrt(np.diag(pcov))
        tauErr = perr[1]
    except RuntimeError:
        print("Could not fit.")
        popt = [np.nan, np.nan, np.nan]
    return popt


def fit_decay2(x, y):
    """
    Function to fit the data, y with a mono-exponential decay using Levenberg-Marquardt algorithm.
    Return fitting parameters [a, tau, c].
    """

    # Guess initial fitting parameters
    t_loc = np.where(y <= y[0] / np.e)[0][0]
    tau = x[t_loc]
    p0 = [max(y), tau, max(y), tau/2, min(y)]

    # Fitting
    try:
        # Fit using Levenberg-Marquardt algorithm
        popt, pcov = curve_fit(decay_fn2, x, y, p0=p0)
        # Error in coefficients to 1 std.
        perr = np.sqrt(np.diag(pcov))
        tauErr = perr[1]
    except RuntimeError:
        print("Could not fit.")
        popt = [np.nan, np.nan, np.nan, np.nan, np.nan]
    return popt


def error_fn(p, x, y):
    """Sum-Squared-Error Cost Function for minimize_fit routine."""
    return sum((y - decay_fn(x, *p))**2)


def minimize_fit(x, y, method='nelder-mead'):
    """
    Fit data using scipy's minimize routine.
    Return fitting parameters [a, tau, c].
    """

    # Guess initial fitting parameters
    t_loc = np.where(y <= y[0] / np.e)[0][0]
    tau = x[t_loc]
    p0 = np.array([max(y), tau, min(y)])

    res = minimize(error_fn, p0, args=(x, y),
                   method=method,
                   options={'xtol': 1e-8, 'disp': False}
                   )
    return res.x
