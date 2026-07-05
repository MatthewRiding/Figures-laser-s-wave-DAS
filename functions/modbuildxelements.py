import numpy as np


def build_x_elements_m(n_tx, pitch_mm):
    n_gaps = n_tx - 1
    aperture_mm = pitch_mm * n_gaps
    # Centre the array around x = 0:
    min_mm = - aperture_mm / 2
    x_elements_mm = min_mm + (np.arange(0, n_tx) * pitch_mm)
    return x_elements_mm * 10**-3
