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
from functions.modconvertvtonm import convert_quartet_v_to_nm
from functions.modcalculatecriticalangle import calculate_critical_angle_degrees

# IPython magic:
%matplotlib qt
%load_ext autoreload
%autoreload 2

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
n_pixels_z = 200
# Wave speeds:
c_l_mpers = 6370
c_t_mpers = 3130
c_lsaw_mpers = 6000

# Measure parameters:
n_elements = np.shape(fmc_3d_nm)[2]
n_samples = np.shape(fmc_3d_nm)[0]
period_us = (t_max_us - t_min_us) / (n_samples - 1)
frequency_sampling_hz = 1 / (period_us * 10**-6)

# Calculate theta_crit:
theta_crit_deg = calculate_critical_angle_degrees(c_l_mpers,
                                                  c_t_mpers)

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


# Define a function to simplify changing wave sets:
def get_tfm_params_Al_SDH(wave_set_string):
    tfm_params = TFMParamsSepMasks('',
                                   pitch_mm,
                                   c_l_mpers,
                                   c_t_mpers,
                                   wave_set_string,
                                   x_width_mm,
                                   z_max_mm,
                                   n_pixels_z)
    return tfm_params


def get_tfm_params_Al_SDH_HT():
    tfm_params_HT = TFMParamsSepMasks('',
                                      pitch_mm,
                                      c_l_mpers,
                                      c_t_mpers,
                                      'H-T',
                                      x_width_mm,
                                      z_max_mm,
                                      n_pixels_z,
                                      mask_tf=True,
                                      mask_angle_deg=theta_crit_deg,
                                      mask_behaviour_string_gen='Ignore below')
    return tfm_params_HT


# Define a function to compute a TFM image:
def compute_tfm_Al_SDH(wave_set_string):
    # Get TFM params for this wave set:
    if wave_set_string == 'H-T':
        tfm_params = get_tfm_params_Al_SDH_HT()
    else:
        tfm_params = get_tfm_params_Al_SDH(wave_set_string)
    # Compute intensity image:
    intensity_image_complex = compute_tfm_separate_masks('', fmc_3d_processed_nm,
                                                         tfm_params, time_vector_us,
                                                         signal_progress)[0]
    # Convert intensity to root-power decibels by normalising relative to a maximum intensity value.
    # To avoid saturation by SAW crosstalk, ignore pixels in the upper third of the image.
    row_index_below_which_to_max = round(tfm_params.n_pixels_z / 3)
    reference_amplitude_0dB = np.max(
        np.abs(intensity_image_complex[row_index_below_which_to_max:, :]))

    # Convert to root-power dB using this reference amplitude:
    image_decibels = 20 * \
        np.log10(np.abs(intensity_image_complex) / reference_amplitude_0dB)
    return intensity_image_complex, image_decibels


# Compute TFM images:
im_complex_LL, im_dB_LL = compute_tfm_Al_SDH('L-L')
im_complex_TT, im_dB_TT = compute_tfm_Al_SDH('T-T')
im_complex_HT, im_dB_HT = compute_tfm_Al_SDH('H-T')
im_complex_hybridS, im_dB_hybridS = compute_tfm_Al_SDH('Hybrid-S')

# Plot TFM images:
fig, (ax_1, ax_2, ax_3, ax_4) = plt.subplots(4, 1, layout='constrained')

v_min = -70
v_max = 70

def plot_displacement_image(ax, im_complex):
    ax.imshow(np.real(im_complex), cmap=cc.m_CET_D7,
              vmin=v_min, vmax=v_max)

ax_1.set_title('L-L')
plot_displacement_image(ax_1, im_complex_LL)

ax_2.set_title('T-T')
plot_displacement_image(ax_2, im_complex_TT)

ax_3.set_title('H-T')
plot_displacement_image(ax_3, im_complex_HT)

ax_4.set_title('Hybrid-S')
plot_displacement_image(ax_4, im_complex_hybridS)
