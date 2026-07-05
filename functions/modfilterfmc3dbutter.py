from scipy.signal import butter, sosfiltfilt


def filter_fmc3d_butter(fmc_3d, frequency_sampling_hertz, butter_order,
                        band_type, Wn_Hz):
    # Apply a Butterworth band pass filter of order 'butter_order' to the A-scans in fmc_3d,
    # with a pass band of (band_min, band_max):

    # First, create the second-order sections (sos) for the butterworth filter:
    sos = butter(butter_order, Wn_Hz, btype=band_type,
                 fs=frequency_sampling_hertz, output='sos')

    # Next, apply the filter to the A-scan matrix using the sosfilt function:
    fmc_3d_filtered = sosfiltfilt(sos, fmc_3d, 0)

    return fmc_3d_filtered


def filter_quartet(
    a_scan,
    f_sampling_hz
):
    """Bandpass filter to reflect the bandwidth of the Sound & Bright quartet
    interferometer (1 - 60 MHz).
    """
    a_scan_filtered = filter_fmc3d_butter(a_scan,
                                          f_sampling_hz,
                                          10,
                                          'bandpass',
                                          [1 * 10**6, 60 * 10**6])
    return a_scan_filtered


def bandpass_fmc3d_butter(fmc_3d, frequency_sampling_hertz, butter_order, band_min_MHz, band_max_MHz):
    # Apply a Butterworth band pass filter of order 'butter_order' to the A-scans in fmc_3d,
    # with a pass band of (band_min, band_max):

    # First, create the second-order sections (sos) for the butterworth filter:
    sos = butter(butter_order, [band_min_MHz * 10**6, band_max_MHz * 10**6],
                 btype='bandpass', fs=frequency_sampling_hertz, output='sos')

    # Next, apply the filter to the A-scan matrix using the sosfilt function:
    fmc_3d_filtered = sosfiltfilt(sos, fmc_3d, 0)

    return fmc_3d_filtered
