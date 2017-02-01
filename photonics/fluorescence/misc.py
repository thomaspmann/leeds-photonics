def filterLifetime(time, intensity, cutoff=10000, order=7):
    """
    Function to apply a low pass filter to a fluorescence signal.

    cutoff: desired cutoff frequency of the filter, Hz
    order: butterworth filter order

    returns
            time: time array, unmodified
            y: the filtered data

    http://tinyurl.com/gqqmsdz
    """

    from scipy.signal import butter, lfilter, freqz
    import matplotlib.pyplot as plt

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # Filter requirements.
    # order = 7
    # fs = int(len(time)/ ((max(time)*1E-3)))       # sample rate, Hz
    fs = int(1 / (0.008176 * 1E-3))
    # cutoff = 10000  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    # Plot the frequency response.
    w, h = freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5 * fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

    # Demonstrate the use of the filter.
    # First make some data to be filtered.
    n = len(time)  # total number of samples
    data = intensity

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(data, cutoff, fs, order)

    plt.subplot(2, 1, 2)
    plt.plot(time, data, 'b-', label='data')
    plt.plot(time, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()

    return time, y
