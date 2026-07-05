import numpy as np


def ricker_signal_unit_max(n_samples, sigma):
    t_vector = np.linspace(-1, 1, n_samples)
    ricker_signal = (0.5 *
                     (2 / (np.sqrt(3 * sigma) * np.pi**(1/4))) *
                     (1 - (t_vector / sigma)**2) *
                     (np.exp(-(t_vector**2 / (2 * sigma**2))))
                     )
    return ricker_signal
