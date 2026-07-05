import matplotlib.pyplot as plt
import colorcet as cc
from scipy.signal import detrend
import numpy as np


from functions.modloadnmoperiodicampsfrommatfile import load_nmo_periodic_amps_from_mat_file
from functions.modextractcmpgathersfromfmc import extract_cmp_from_fmc2d
from functions.modextractmonostaticsaft import extract_mss_from_fmc2d
from functions.modfilterfmc3dbutter import filter_fmc3d_butter
from functions.modsdhcorrect import sdh_correct_t_0_based
from functions.modgeneratetimevectorus import generate_time_vector_us
from functions.modcalculatethetacrit import calculate_theta_crit_deg


# Load the data (mon-static saft (mss) scan):
mss_2d_raw = load_nmo_periodic_amps_from_mat_file(
    r"C:\Users\mattr\OneDrive - University of Strathclyde\General\00_Experimental Data\2025_04_02_Matt_mild_steel_cal_block\Target 2 top 1mm SDH from arc\Scan 1 wfm 29\FMC.mat")

# Detrend each A-scan:
mss_2d_detrend = detrend(mss_2d_raw, axis=0, type='constant')

# Create axis vectors:
t_min_us = -0.25
t_max_us = 4.74
n_samples = np.shape(mss_2d_detrend)[0]
time_vector_us = generate_time_vector_us(t_min_us, t_max_us, n_samples)
pitch_mm = 0.05
n_tx = np.shape(mss_2d_detrend)[1]
aperture_mm = pitch_mm * (n_tx - 1)
x_elements_mm = np.linspace(0, aperture_mm, n_tx)

# Plot:
c_map = cc.m_CET_D7
c_max_mV = 10
c_min_mV = -10

# Filter:
period_s = ((t_max_us + t_min_us) * 10**-6) / (n_samples-1)
frequency_sampling = 1 / period_s
Wn_MHz = 2
proportion_to_trim_from_start = 0.15
row_start = round(n_samples * proportion_to_trim_from_start)
mss_filtered = filter_fmc3d_butter(mss_2d_detrend[row_start:, :],
                                   frequency_sampling, 10,
                                   'highpass', Wn_MHz)
# Adjust time axis to reflect trimming:
t_min_trimmed_us = t_min_us + ((t_max_us - t_min_us)
                               * proportion_to_trim_from_start)
# Created corrected section:
x_sdh_mm = 6.25
z_sdh_centre_mm = 2
radius_sdh_mm = 0.5
c_T_mpers = 3240
c_L_mpers = 5918
tan_theta_crit = c_T_mpers / np.sqrt(c_L_mpers**2 - c_T_mpers**2)
t_0_min_sdh_us = 0.5
t_0_max_sdh_us = 1
n_t_0s = 300
(mss_corr, t_0_vector_us) = sdh_correct_t_0_based(mss_filtered, t_0_min_sdh_us,
                                        t_0_max_sdh_us, n_t_0s, x_elements_mm,
                                        time_vector_us[row_start:], c_T_mpers,
                                        x_sdh_mm, radius_sdh_mm)

# Calculate x positions corresponding to angles of interest:
theta_crit_deg = calculate_theta_crit_deg(c_L_mpers, c_T_mpers)
x_crit_mm = x_sdh_mm + (z_sdh_centre_mm * tan_theta_crit)
x_45_mm = x_sdh_mm + 2

# Make MSS figure:
fig, (ax_3, ax_4) = plt.subplots(2, 1)

im_3 = ax_3.imshow(mss_filtered*1000, cmap=c_map, aspect='auto',
                   extent=(0, aperture_mm,
                           t_max_us, t_min_trimmed_us))
ax_3.set_xlabel('x (mm)')
ax_3.set_ylabel(r'Time ($\mu$s)')
im_3.set_clim(vmin=-10, vmax=10)

im_4 = ax_4.imshow(mss_corr*1000, cmap=c_map, aspect='auto',
                   extent=(0, aperture_mm,
                           t_0_max_sdh_us, t_0_min_sdh_us))
ax_4.set_xlabel('x (mm)')
ax_4.set_ylabel(r'Time ($\mu$s)')
im_4.set_clim(vmin=-10, vmax=10)
ax_4.axvline(x_crit_mm, ls='dashed', color='w')
ax_4.axvline(x_45_mm, ls='dashed', color='k')
