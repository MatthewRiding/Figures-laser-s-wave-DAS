from scipy.signal import detrend


def detrend_fmc_3d(fmc_3d):
    # Subtract the mean using scipy detrend in 'constant' mode:
    fmc_3d_detrend = detrend(fmc_3d, axis=0, type='constant')
    return fmc_3d_detrend
