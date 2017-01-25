import numpy as np
from scipy.optimize import curve_fit, minimize


# Helper Functions
def decay(t, a, tau, c):
    """
    Mono-exponential fitting function. t is the time.
    """
    return a * np.exp(-t / tau) + c


def fit_decay(x, y, reject=0):
    """
    Function to fit the data, y with a mono-exponential decay using Levenberg-Marquardt algorithm.
    Return fitting parameters [a, tau, c].
    """
    # Reject Time
    ind = np.where(x >= reject)
    x = x[ind]
    y = y[ind]

    # Subtract baseline noise
    y -= min(y)

    # Normalise
    y /= y[0]

    # Guess initial fitting parameters
    t_loc = np.where(y <= 1 / np.e)
    p0 = [1, x[t_loc[0][0]], 0]

    # Fitting
    try:
        # Fit using Levenberg-Marquardt algorithm
        popt, pcov = curve_fit(decay, x, y, p0=p0)
        # Error in coefficients to 1 std.
        perr = np.sqrt(np.diag(pcov))
        tauErr = perr[1]
    except RuntimeError:
        print("Could not fit.")
        popt = [np.nan, np.nan, np.nan]
    return popt


