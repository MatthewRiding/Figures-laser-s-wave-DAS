import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc

from functions.modloadfmclpfrommatfile import load_fmclp_from_mat_file
from functions.modcomputetfmcomplexgdmask import compute_tfm_complex_gd_masks_same
from functions.modcomputetfmseparategdmasks import compute_tfm_separate_masks
from classdefs.modfakesignal import FakeSignal
from classdefs.modtfmparams import TFMParams
from classdefs.modtfmparamsseparatemasks import TFMParamsSepMasks
from functions.moddetrendfmc3d import detrend_fmc_3d
from functions.modfilterfmc3dbutter import filter_fmc3d_butter
from corevariables.modieeeplotting import line_width_plots
from functions.modthetacrittrig import calculate_theta_crit_deg
from functions.modconvertvtonm import convert_quartet_v_to_nm

# IPython magic:
# %matplotlib qt
# %load_ext autoreload
# %autoreload 2

# Load Al SDH FMC:
mat_file_path_string = r"C:\Users\mattr\OneDrive - University of Strathclyde\General\00_Experimental Data\2025_02_05_Matt_SDH_polarity_Al_FMC\Scan 2 long\FMC_v7.mat"
fmc_3d_v = load_fmclp_from_mat_file(mat_file_path_string)
fmc_3d_nm = convert_quartet_v_to_nm(fmc_3d_v)

# Input parameters:
# Pitch:
pitch_mm = 0.2
# Time limits:
t_min_us = -0.5
t_max_us = 9.49
# TFM grid:
x_width_mm = 20
z_max_mm = 10
n_pixels_z = 50
# Wave speeds:
c_l_mpers = 6370
c_t_mpers = 3130

# Measure parameters:
n_elements = np.shape(fmc_3d_nm)[2]
n_samples = np.shape(fmc_3d_nm)[0]
period_us = (t_max_us - t_min_us) / (n_samples - 1)
frequency_sampling_hz = 1 / (period_us * 10**-6)

theta_crit_deg = calculate_theta_crit_deg(c_l_mpers, c_t_mpers)

# Create time vector:
time_vector_us = np.linspace(t_min_us, t_max_us, n_samples, endpoint=True)

# Apply detrend and filters to fmc_3d:
fmc_3d_detrend_nm = detrend_fmc_3d(fmc_3d_nm)
fmc_3d_processed_nm = filter_fmc3d_butter(fmc_3d_detrend_nm,
                                          frequency_sampling_hz,
                                          10, 'bandpass',
                                          (2.5*10**6, 14*10**6))
# fmc_3d_processed = fmc_3d_detrend

# Instantiate a fake signal:
signal_progress = FakeSignal()

# Define functions for use in the loop:


# def get_tfm_params(wave_set_string,
#                    mask_tf=False,
#                    mask_angle_deg=90,
#                    mask_behaviour_string='Ignore above'
#                    ):
#     tfm_params = TFMParams('',
#                            pitch_mm,
#                            c_l_mpers,
#                            c_t_mpers,
#                            wave_set_string,
#                            x_width_mm,
#                            z_max_mm,
#                            n_pixels_z,
#                            False,
#                            10,
#                            1,
#                            12,
#                            mask_tf,
#                            mask_angle_deg,
#                            mask_behaviour_string)
#     return tfm_params


def get_tfm_params_ht(mask_behaviour_string_gen='Ignore below',
                      mask_behaviour_string_det='Ignore above'
                      ):
    tfm_params = TFMParamsSepMasks('',
                                   pitch_mm,
                                   c_l_mpers,
                                   c_t_mpers,
                                   'H-T',
                                   x_width_mm,
                                   z_max_mm,
                                   n_pixels_z,
                                   False,
                                   10,
                                   1,
                                   12,
                                   True,
                                   theta_crit_deg,
                                   mask_behaviour_string_gen=mask_behaviour_string_gen,
                                   mask_behaviour_string_det=mask_behaviour_string_det)
    return tfm_params


# def compute_tfm_from_params(tfm_params):
#     intensity_image_complex = compute_tfm_complex_gd_masks_same('', fmc_3d_processed_nm,
#                                                                 tfm_params, time_vector_us,
#                                                                 signal_progress)[0]
#     # Convert intensity to root-power decibels by normalising relative to a maximum intensity value.
#     # To avoid saturation by SAW crosstalk, ignore pixels in the upper third of the image.
#     row_index_below_which_to_max = round(tfm_params.n_pixels_z / 3)
#     reference_amplitude_0dB = np.max(
#         np.abs(intensity_image_complex[row_index_below_which_to_max:, :]))

#     # Convert to root-power dB using this reference amplitude:
#     image_decibels = 20 * \
#         np.log10(np.abs(intensity_image_complex) / reference_amplitude_0dB)
#     return intensity_image_complex, image_decibels


def compute_tfm_from_params_sep_masks(tfm_params_sep_masks):
    intensity_image_complex = compute_tfm_separate_masks('', fmc_3d_processed_nm,
                                                         tfm_params_sep_masks, time_vector_us,
                                                         signal_progress)[0]
    # Convert intensity to root-power decibels by normalising relative to a maximum intensity value.
    # To avoid saturation by SAW crosstalk, ignore pixels in the upper third of the image.
    row_index_below_which_to_max = round(tfm_params_sep_masks.n_pixels_z / 3)
    reference_amplitude_0dB = np.max(
        np.abs(intensity_image_complex[row_index_below_which_to_max:, :]))

    # Convert to root-power dB using this reference amplitude:
    image_decibels = 20 * \
        np.log10(np.abs(intensity_image_complex) / reference_amplitude_0dB)
    return intensity_image_complex, image_decibels


# Ok, so let's get a head wave image (using ignore below the critical angle):
# Let's compare H-T with an 'ignore below' mask, versus HT just using the travel time function
# (which currently has a built-in mask):
# tfm_params_both_masks_below = get_tfm_params(
#     'H-T', True, theta_crit_deg, 'Ignore below')
tfm_params_ht_sep_masks = get_tfm_params_ht()

# intensity_image_complex_1, image_decibels_1 = compute_tfm_from_params(
#     tfm_params_both_masks_below)
intensity_image_complex_2, image_decibels_2 = compute_tfm_from_params_sep_masks(
    tfm_params_ht_sep_masks)


# # Plotting:
# fig, ((ax_1, ax_2), (ax_3, ax_4)) = plt.subplots(2, 2, layout='constrained')

# ax_1.set_title('Same \'Ignore below\' mask for gen and det')
# ax_1.imshow(np.real(intensity_image_complex_1), cmap=cc.m_CET_D7)
# ax_2.imshow(image_decibels_1)

# ax_3.set_title('Gen mask below, det mask above')
# ax_3.imshow(np.real(intensity_image_complex_2), cmap=cc.m_CET_D7)
# ax_4.imshow(image_decibels_2)
