import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .spectrometer import decay_fn, decay_fn2


def fit_decay(x, y, p0=None, print_out=True):
    """
    Function to fit data with a single-exp. decay using Levenberg-Marquardt algorithm.
    :param x: time array
    :param y: intensity array
    :param p0: (Optional) initial guess for fitting parameters [a, tau, c].
    :return: fluorescence parameters popt [a, tau, c]
    """
    # Guess initial fluorescence parameters
    if p0 is None:
        t_loc = np.where(y <= y[0] / np.e)[0][0]
        tau = x[t_loc]
        p0 = [max(y), tau, min(y)]

    # Fitting
    try:
        # Fit using Levenberg-Marquardt algorithm
        popt, pcov = curve_fit(decay_fn, x, y, p0=p0)
        # Error in coefficients to 1 std.
        perr = np.sqrt(np.diag(pcov))

        # Print results in a table
        if print_out:
            headers = ["A", "tau", "C"]
            a = u"{0:.2f}\u00B1{1:.2f}".format(popt[0], perr[0])
            tau = u"{0:.2f}\u00B1{1:.2f}".format(popt[1], perr[1])
            c = u"{0:.2f}\u00B1{1:.2f}".format(popt[2], perr[2])
            table = [[a, tau, c]]
            print(tabulate(table, headers=headers))

    except RuntimeError:
        print("Could not fit.")
        popt = [np.nan, np.nan, np.nan]
        perr = [np.nan, np.nan, np.nan]
    return popt, perr


def fit_decay2(x, y, p0=None, print_out=True):
    """
    Function to fit data with a double-exp. decay using Levenberg-Marquardt algorithm.
    :param x: time array
    :param y: intensity array
    :param fn: decay function to fit
    :return: fluorescence parameters popt [a1, tau1, a2, tau2, c]
    """
    # Guess initial fluorescence parameters
    if p0 is None:
        t_loc = np.where(y <= y[0] / np.e)[0][0]
        tau = x[t_loc]
        p0 = [max(y), tau / 2, max(y), tau, min(y)]

    # Fitting
    try:
        # Fit using Levenberg-Marquardt algorithm
        popt, pcov = curve_fit(decay_fn2, x, y, p0=p0)
        # Error in coefficients to 1 std.
        perr = np.sqrt(np.diag(pcov))

        # Print results in a table
        if print_out:
            headers = ["A_1", "tau_1", "A_2", "tau_2", "C", "Chisq"]
            a1 = u"{0:.2f}\u00B1{1:.2f}".format(popt[0], perr[0])
            tau1 = u"{0:.2f}\u00B1{1:.2f}".format(popt[1], perr[1])
            a2 = u"{0:.2f}\u00B1{1:.2f}".format(popt[2], perr[2])
            tau2 = u"{0:.2f}\u00B1{1:.2f}".format(popt[3], perr[3])
            c = u"{0:.2f}\u00B1{1:.2f}".format(popt[4], perr[4])
            table = [[a1, tau1, a2, tau2, c]]
            print(tabulate(table, headers=headers))

    except RuntimeError:
        print("Could not fit.")
        popt = [np.nan, np.nan, np.nan, np.nan, np.nan]
        perr = [np.nan, np.nan, np.nan, np.nan, np.nan]
    return popt, perr


def plot_decay(x, y, fn, popt, log=True, norm=False):
    """
    Plot a fluorescence decay with the resiuals and chisq value of the fit.
    :param x: x data
    :param y: y data
    :param fn: function to fit
    :param popt: parameters to the function
    :param log: (bool) whether to use a log plot for the decay
    :param norm: (bool) Normlaise the output graph
    :return: fig handle
    """
    residuals = y - fn(x, *popt)
    y_pred = fn(x, *popt)
    if norm:
        ref = y[0]
        y /= ref
        y_pred /= ref

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.set_ylabel('Intensity (A.U.)')
    ax1.plot(x, y, label="Original Noised Data")
    ax1.plot(x, y_pred, label="Fitted Curve")
    ax1.legend()
    if log:
        ax1.set_yscale('log')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Residuals')
    ax2.plot(x, residuals)
    ax2.axhline(0, color='k')
    plt.tight_layout()
    plt.show()
    return fig


def rise_fn(t, a, tau, c):
    """Single-exponential rise fluorescence function. t is the time."""
    return a - a * np.exp(-t / tau) + c


def fit_rise(x, y, p0=None):
    """
    Function to fit data with a single-exp. decay using Levenberg-Marquardt algorithm.
    :param x: time array
    :param y: intensity array
    :param p0: (Optional) initial guess for fitting parameters [a, tau, c].
    :return: fluorescence parameters popt [a, tau, c]
    """
    # Guess initial fluorescence parameters
    if p0 is None:
        t_loc = np.where(y <= y[-1] / np.e)[0][0]
        tau = x[t_loc]
        p0 = [max(y), tau, min(y)]

    # Fitting
    try:
        # Fit using Levenberg-Marquardt algorithm
        popt, pcov = curve_fit(rise_fn, x, y, p0=p0)
        # Error in coefficients to 1 std.
        perr = np.sqrt(np.diag(pcov))

    except RuntimeError:
        print("Could not fit.")
        popt = [np.nan, np.nan, np.nan]
        perr = [np.nan, np.nan, np.nan]
    return popt, perr


def rise_fn(t, a, tau, c):
    """Single-exponential rise fluorescence function. t is the time."""
    return a - a * np.exp(-t / tau) + c


def fit_rise(x, y, p0=None, print_out=True):
    """
    Function to fit data with a single-exp. decay using Levenberg-Marquardt algorithm.
    :param x: time array
    :param y: intensity array
    :param p0: (Optional) initial guess for fitting parameters [a, tau, c].
    :return: fluorescence parameters popt [a, tau, c]
    """
    # Guess initial fluorescence parameters
    if p0 is None:
        t_loc = np.where(y <= y[-1] / np.e)[0][0]
        tau = x[t_loc]
        p0 = [max(y), tau, min(y)]

    # Fitting
    try:
        # Fit using Levenberg-Marquardt algorithm
        popt, pcov = curve_fit(rise_fn, x, y, p0=p0)
        # Error in coefficients to 1 std.
        perr = np.sqrt(np.diag(pcov))

    except RuntimeError:
        print("Could not fit.")
        popt = [np.nan, np.nan, np.nan]
        perr = [np.nan, np.nan, np.nan]
        chisq = np.nan
    return popt, perr
