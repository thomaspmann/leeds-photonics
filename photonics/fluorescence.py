import numpy as np
from scipy.optimize import curve_fit, minimize


def shift_time(x, length):
    """Shift time axis to the left by length. Used to account for pump"""
    x -= length
    return x


def reject_time(x, y, reject_start=0, reject_end=0):
    """Reject x,y data before 'reject_start' or after 'reject_end' time """
    ind = np.where((x >= reject_start) & (x < x[-1] - reject_end))
    return x[ind], y[ind]


def normalise(y, ref="max"):
    """Normalise array y with respect to the ref point (either 'max' or 'start')"""
    # Subtract baseline noise
    y -= min(y)
    # Normalise
    if ref == "max":
        y /= max(y)
    elif ref == "start":
        y /= y[0]
    else:
        raise ValueError("point option must be either 'max' or 'start'.")
    return y


def decay_fn(t, a, tau, c):
    """Single-exponential decay fluorescence function. t is the time."""
    return a * np.exp(-t / tau) + c


def fit_decay(x, y):
    """
    Function to fit data with a single-exp. decay using Levenberg-Marquardt algorithm.
    :param x: time array
    :param y: intensity array
    :param fn: decay function to fit
    :return: fluorescence parameters popt [a, tau, c]
    """
    # Guess initial fluorescence parameters
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


def decay_fn2(t, a1, tau1, a2, tau2, c):
    """Double-exponential decay fluorescence function. t is the time."""
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c


def fit_decay2(x, y):
    """
    Function to fit data with a double-exp. decay using Levenberg-Marquardt algorithm.
    :param x: time array
    :param y: intensity array
    :param fn: decay function to fit
    :return: fluorescence parameters popt [a1, tau1, a2, tau2, c]
    """
    # Guess initial fluorescence parameters
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


def error_fn(p, x, y, fn=decay_fn):
    """
    Sum-Squared-Error Cost Function for minimize_fit routine.
    :param p: fn params
    :param x: x data
    :param y: y data
    :param fn: function to fit
    :return: Sum-Squared-Error
    """
    return sum((y - fn(x, *p))**2)


def minimize_fit(x, y, method='nelder-mead'):
    """
    Fit data using scipy's minimize routine.
    Return fluorescence parameters [a, tau, c].
    """

    # Guess initial fluorescence parameters
    t_loc = np.where(y <= y[0] / np.e)[0][0]
    tau = x[t_loc]
    p0 = np.array([max(y), tau, min(y)])

    res = minimize(error_fn, p0, args=(x, y),
                   method=method,
                   options={'xtol': 1e-8, 'disp': False}
                   )
    return res.x


def chi2(x, y, fn, popt):
    """
    Normalised chi-squared value.
    For derivation see pgs. 19-20 "Topics in Fluorescence Spectroscopy, Volume 1" by Lakowicz.
    :param x: x data
    :param y: y data
    :param fn: fitted function
    :param popt: parameters for fitted function
    :return: Normalised chi-squared value
    """
    residuals = y - fn(x, *popt)
    std = np.std(residuals)
    return sum((residuals / std) ** 2) / (len(y) - len(popt))


def plot_decay(x, y, fn, popt, log=True):
    """
    Plot a fluorescence decay with the resiuals and chisq value of the fit.
    :param x: x data
    :param y: y data
    :param fn: function to fit
    :param popt: parameters to the function
    :param log: (bool) whether to use a log plot for the decay
    :return: fig handle
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.set_ylabel('Intensity (A.U.)')
    ax1.plot(x, y, '.', alpha=0.6, label="Original Noised Data")
    ax1.plot(x, fn(x, *popt), label="Fitted Curve")
    ax1.legend()
    if log:
        ax1.set_yscale('log')

    residuals = y - fn(x, *popt)
    # std = np.std(residuals)
    ax2.set_ylabel('Residuals (Std. Dev)')
    ax2.plot(x, residuals / np.std(residuals))
    ax2.axhline(0, color='k')
    ax2.set_xlabel('Time (ms)')

    chisq = chi2(x, y, fn, popt)
    ax1.set_title('Chisq = {0:.3f}'.format(chisq))
    plt.show()
    return fig


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
