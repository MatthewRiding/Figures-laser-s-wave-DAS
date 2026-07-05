import numpy as np


def nmo_correct(b_scan_ds, time_vector_us,
                t_0_echo_us, t_0_half_width_us, n_t_0s,
                x_vector_mm, wave_speed):
    # Generate vector of t_0 values to evaluate delay laws for:
    t_0_min_us = t_0_echo_us - t_0_half_width_us
    t_0_max_us = t_0_echo_us + t_0_half_width_us
    t_0_vector_us = np.linspace(t_0_min_us, t_0_max_us, n_t_0s)

    # Pre-allocate output array:
    n_A_scans = len(x_vector_mm)
    b_scan_ds_nmo_corrected = np.zeros((n_t_0s, n_A_scans))

    # Iterate over A-scans:
    for i, (A_scan_ds, x_mm) in enumerate(zip(b_scan_ds.T, x_vector_mm)):
        t_hyp_s = np.sqrt(((x_mm * 10**-3)**2) / (wave_speed**2)
                          + (t_0_vector_us * 10**-6)**2)
        b_scan_ds_nmo_corrected[:, i] = np.interp(t_hyp_s,
                                                  time_vector_us * 10**-6,
                                                  A_scan_ds, 0, 0)

    return b_scan_ds_nmo_corrected, t_0_vector_us


def nmo_correct_b_vector(
    b_scan_ds,
    time_vector_us,
    b_vector_mm,
    x_vector_mm,
    wave_speed_mpers
):
    """Assumes no mode conversion: wave speed on pre-reflection
    leg = wave speed on post reflection leg.
    """

    # Convert vector of sample thickness values (b) into
    # vector of t_0 values to evaluate delay laws for:
    t_0_vector_s = (b_vector_mm * 2 * 10**-3) / wave_speed_mpers

    # Pre-allocate output array:
    n_t_0s = len(t_0_vector_s)
    n_A_scans = len(x_vector_mm)
    b_scan_ds_nmo_corrected = np.zeros((n_t_0s, n_A_scans))

    # Iterate over A-scans:
    for i, (A_scan_ds, x_mm) in enumerate(zip(b_scan_ds.T, x_vector_mm)):
        t_hyp_s = np.sqrt(((x_mm * 10**-3)**2) / (wave_speed_mpers**2)
                          + (t_0_vector_s)**2)
        b_scan_ds_nmo_corrected[:, i] = np.interp(t_hyp_s,
                                                  time_vector_us * 10**-6,
                                                  A_scan_ds, 0, 0)

    return b_scan_ds_nmo_corrected
