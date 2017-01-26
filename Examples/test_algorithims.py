import numpy as np
import lifetime.decay as lm
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_data(p_init, std=.0):
    """Create decay curve with WGN noise of standard deviation std times the amplitude."""
    x = np.linspace(0, 10*tau, num=1000)
    y = lm.decay_fn(x, *p_init)

    # Add noise to signal. Std of % amplitude
    if std != 0:
        y += np.random.normal(0, std*max(y), 1000)
    return x, y


if __name__ == "__main__":

    a = 1             # Amplitude
    tau = 10          # Lifetime in ms
    c = 0             # Noise offset
    p_init = [a, tau, c]

    tau_list = []
    for i in tqdm(range(1000)):
        x, y = create_data(p_init, std=0.1)
        x, y = lm.prepare_data(x, y, reject_start=0)

        p = lm.fit_decay(x, y)
        # p = lm.minimize_fit(x, y)
        tau = p[1]
        tau_list.append(tau)

    print(np.mean(tau_list), np.std(tau_list))

    # plt.figure()
    # plt.plot(x, y, label='Noised data')
    # plt.plot(x, lm.decay_fn(x, *p_init), label='Un-noised data')
    # plt.plot(x, lm.decay_fn(x, *p), label='Fit')
    # plt.legend()
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Intensity (A.U.)')
