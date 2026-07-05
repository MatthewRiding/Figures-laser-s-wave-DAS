import numpy as np
from scipy.signal import hilbert


def slice_b_scan_for_angular_range(b_scan_ds_corr, angles_vector_deg,
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
    return b_scan_ds_slice


def slice_b_scan_pos_and_neg_angles(b_scan_ds_corr, angles_vector_deg,
                                    theta_min_deg, theta_max_deg):
    # Convert start and max angles into indices to slice the B_scan array:
    # Here, the angles vector will feature both positive and negative angles
    # either side of zero:

    # Positive range (i.e. from +theta_start to +theta_max):
    idx_start_pos = (np.abs(angles_vector_deg - theta_min_deg)).argmin()
    idx_stop_pos = (np.abs(angles_vector_deg - theta_max_deg)).argmin()
    b_scan_ds_slice_pos = b_scan_ds_corr[:, idx_start_pos:idx_stop_pos]

    # Negative range (i.e. from -theta_start to -theta_max):
    # Indexing from left to right, idx_start will be dictated by theta_max:
    idx_start_neg = (np.abs(angles_vector_deg + theta_max_deg)).argmin()
    idx_stop_neg = (np.abs(angles_vector_deg + theta_min_deg)).argmin()
    b_scan_ds_slice_neg = b_scan_ds_corr[:, idx_start_neg:idx_stop_neg]

    return b_scan_ds_slice_pos, b_scan_ds_slice_neg


def compute_vcf_along_rows(b_scan_ds_corr):
    # Compute the vector coherence factor along each row:
    b_scan_complex = hilbert(b_scan_ds_corr, axis=0)
    b_scan_complex_units = (b_scan_complex /
                            np.abs(b_scan_complex))
    n_samples = np.shape(b_scan_complex_units)[1]
    vcf_waveform = ((1 / n_samples) *
                    np.abs(np.sum(b_scan_complex_units, axis=1)))
    return vcf_waveform


def vcf_gather(b_scan_ds_corr, angles_vector_deg,
               theta_start_deg=None, theta_max_deg=None):
    b_scan_ds_slice = slice_b_scan_for_angular_range(b_scan_ds_corr,
                                                     angles_vector_deg,
                                                     theta_start_deg,
                                                     theta_max_deg)
    vcf_waveform = compute_vcf_along_rows(b_scan_ds_slice)
    return vcf_waveform


def vcf_gather_pos_and_neg_angles(b_scan_ds_corr, angles_vector_deg,
                                  theta_start_deg=None, theta_max_deg=None):
    # We only need to slice if the user has specified an angular range:
    if theta_max_deg:
        # Slice B-scan to extract positive and negative ranges:
        (b_scan_ds_slice_pos,
         b_scan_ds_slice_neg) = slice_b_scan_pos_and_neg_angles(b_scan_ds_corr,
                                                                angles_vector_deg,
                                                                theta_start_deg,
                                                                theta_max_deg)
        # H-stack the two arrays:
        stack = np.hstack((b_scan_ds_slice_pos, b_scan_ds_slice_neg))
    else:
        stack = b_scan_ds_corr

    vcf_waveform = compute_vcf_along_rows(stack)

    return vcf_waveform
