import numpy as np


def extract_mss_from_fmc2d(fmc_2d):
    # Measure values from dimensions of fmc_2d:
    n_samples, n_A_scans = np.shape(fmc_2d)
    n_tx = int(np.sqrt(n_A_scans))

    # Index  using striding:
    col_start = 0
    step_col = n_tx + 1
    mss = fmc_2d[:, col_start:None:step_col]

    return mss
