import numpy as np

from functions.modbt0 import calculate_t_0_HS_us_from_b_mm


def cdp_head_wave_correct(
    b_scan_ds,
    b_vector_mm,
    x_vector_mm,
    time_vector_us,
    c_t_mpers,
    c_LSW_mpers
):
    """CDP = common detection point, referring to the scan pattern.
    Assumes reflection from a horiontal free surface (HFS).

    :param: t_0_HS_vector_us: The straight line time-axis intercepts
    for the head wave (units: microseconds [us]).
    """
    # Convert b (thickness) vector to t_0_HS vector:
    t_0_HS_vector_us = calculate_t_0_HS_us_from_b_mm(b_vector_mm,
                                                     c_t_mpers,
                                                     c_LSW_mpers)

    # Pre-allocate output array:
    n_t_0s = len(t_0_HS_vector_us)
    n_A_scans = len(x_vector_mm)
    b_scan_ds_hw_corrected = np.zeros((n_t_0s, n_A_scans))

    # Iterate over A-scans:
    for i, (A_scan_ds, x_mm) in enumerate(zip(b_scan_ds.T, x_vector_mm)):
        t_delay_s = (((x_mm * 10**-3) / c_LSW_mpers) +
                     (t_0_HS_vector_us * 10**-6))
        b_scan_ds_hw_corrected[:, i] = np.interp(t_delay_s,
                                                 time_vector_us * 10**-6,
                                                 A_scan_ds, 0, 0)

    return b_scan_ds_hw_corrected
