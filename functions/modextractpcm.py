import numpy as np
from scipy.signal import hilbert


def extract_pcm(fmc_3d, time_vector_us, delay_matrix_s):
    """Samples a 3D full matrix of A-scans using a 2d matrix of delays (one delay per A-scan) to extract a 2D matrix
    of complex amplitude values that would be summed to a given pixel in a TFM image (the pixel contribution matrix (
    PCM))."""

    # Measure n_tx:
    n_tx = np.shape(delay_matrix_s)[1]

    # Convert s to us:
    time_vector_s = time_vector_us * 10 ** -6

    # Pre-allocate output PCM- this will be an array of complex numbers:
    pcm = np.zeros([n_tx, n_tx], dtype=np.complex128)

    for det_index in range(n_tx):
        for gen_index in range(n_tx):
            # Extract A-scan [i,j]:
            a_scan_amplitudes = hilbert(fmc_3d[:, det_index, gen_index])
            # Extract delay [i,j]:
            delay_s = delay_matrix_s[det_index, gen_index]
            # Interpolate A-scan at delay time & save into PCM:
            pcm[det_index, gen_index] = np.interp(delay_s, time_vector_s, a_scan_amplitudes, left=0, right=0)

    # Since the numpy.interp() function does not propagate the mask from delay_matrix_s (if it is a
    # MaskedArray), we have to manually re-apply the mask:
    mask = np.ma.getmask(delay_matrix_s)
    pcm_processed = np.ma.masked_where(mask, pcm, copy=False)

    return pcm_processed
