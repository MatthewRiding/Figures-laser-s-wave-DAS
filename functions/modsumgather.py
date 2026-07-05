import numpy as np


def sum_gather(b_scan_ds_corr, angles_vector_deg,
               theta_start_deg=None, theta_max_deg=None):
    # Convert start and max angles into indices to slice the B_scan array:
    if theta_start_deg is None:
        idx_start = None
    else:
        idx_start = (np.abs(angles_vector_deg - theta_start_deg)).argmin()

    if theta_max_deg is None:
        idx_max = None
    else:
        idx_max = (np.abs(angles_vector_deg - theta_max_deg)).argmin()

    b_scan_ds_slice = b_scan_ds_corr[:, idx_start:idx_max]
    sum_ds = np.sum(b_scan_ds_slice, axis=1)
    return sum_ds


def sum_gather_pos_and_neg_angles(b_scan_ds_corr, angles_vector_deg,
                                  theta_min_deg, theta_max_deg,
                                  print_indices=False):
    # Convert start and max angles into indices to slice the B_scan array:
    # Here, the angles vector will feature both positive and negative angles
    # either side of zero:

    # Positive range (i.e. from +theta_start to +theta_max):
    idx_start_pos = (np.abs(angles_vector_deg - theta_min_deg)).argmin()
    idx_stop_pos = (np.abs(angles_vector_deg - theta_max_deg)).argmin()
    b_scan_ds_slice_pos = b_scan_ds_corr[:, idx_start_pos:idx_stop_pos]
    sum_ds_pos = np.sum(b_scan_ds_slice_pos, axis=1)

    # Negative range (i.e. from -theta_start to -theta_max):
    # Indexing from left to right, idx_start will be dictated by theta_max:
    idx_start_neg = (np.abs(angles_vector_deg + theta_max_deg)).argmin()
    idx_stop_neg = (np.abs(angles_vector_deg + theta_min_deg)).argmin()
    b_scan_ds_slice_neg = b_scan_ds_corr[:, idx_start_neg:idx_stop_neg]
    sum_ds_neg = np.sum(b_scan_ds_slice_neg, axis=1)

    if print_indices:
        n_columns = np.shape(b_scan_ds_corr)[1]
        print(f'Total number of columns in input B-scan = {n_columns}')
        print(f'Positive: idx start = {idx_start_pos}, idx stop = {idx_stop_pos}')
        print(f'Negative: idx start = {idx_start_neg}, idx stop = {idx_stop_neg}')

    # Sum both sums (from the positive range and negative range):
    sum_ds_final = sum_ds_pos + sum_ds_neg
    return sum_ds_final
