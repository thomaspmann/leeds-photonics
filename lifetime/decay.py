import numpy as np
from scipy.optimize import curve_fit, minimize


def decay_fn(t, a, tau, c):
    """Mono-exponential fitting function. t is the time."""
    return a * np.exp(-t / tau) + c


def error_fn(p, x, y):
    """Sum-Squared-Error Cost Function for minimize_fit routine."""
    return sum((y - decay_fn(x, *p))**2)


def prepare_data(x, y, reject_start=0, reject_end=0):
    """Prepare data before fitting."""

    # Subtract baseline noise
    y -= min(y)

    # Normalise
    y /= y[0]

    # Reject Time
    ind = np.where((x >= reject_start) & (x < x[-1] - reject_end))
    x = x[ind]
    y = y[ind]

    return x, y


def fit_decay(x, y):
    """
    Function to fit the data, y with a mono-exponential decay using Levenberg-Marquardt algorithm.
    Return fitting parameters [a, tau, c].
    """

    # Guess initial fitting parameters
    t_loc = np.where(y <= y[0] / np.e)[0][0]
    tau = x[t_loc]
    p0 = [1, tau, 0]

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
