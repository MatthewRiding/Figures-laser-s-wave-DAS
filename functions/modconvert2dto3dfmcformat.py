import numpy as np


def convert_2d_to_3d_fmclp_format(fmclp_2d):
    # The fmclp in 2d-arry format will have a number of columns equal to n_tx^2:
    n_samples = np.shape(fmclp_2d)[0]
    n_a_scans = np.shape(fmclp_2d)[1]
    n_tx = int(np.sqrt(n_a_scans))

    # Re-shape the 2d-arry into the equivalent fmclp in 3d-array format:
    array_3d = np.reshape(fmclp_2d, (n_samples, n_tx, n_tx))

    return array_3d
