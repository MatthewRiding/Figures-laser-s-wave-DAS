import numpy as np


def extract_cmp_from_fmc2d(fmc_2d, type):
    # Measure values from dimensions of fmc_2d:
    n_samples, n_A_scans = np.shape(fmc_2d)
    n_tx = int(np.sqrt(n_A_scans))

    # Index  using striding:
    if type == 'gen_high':
        col_start = n_tx - 1
        col_stop = round(n_A_scans / 2)
    elif type == 'gen_low':
        col_start = int((np.floor(n_tx/2) * n_tx) + (np.ceil(n_tx/2) - 1))
        col_stop = n_A_scans - 1
    else:
        raise ValueError
    step_col = n_tx - 1
    cmp = fmc_2d[:, col_start:col_stop:step_col]

    return cmp
