import matplotlib.pyplot as plt
import colorcet as cc
import numpy as np

from functions.modloadnmoperiodicampsfrommatfile import load_nmo_periodic_amps_from_mat_file
from functions.modconvertvtonm import convert_quartet_v_to_nm
from classdefs.modparamsets import sdh_fmc_param_set
from NMO_polarity_aluminium import get_params_aluminium_NMO_scan


# Load the data:

# SDH monostatic scan:
fmc_2d_SDH_raw_v = load_nmo_periodic_amps_from_mat_file(
    r"C:\Users\mattr\OneDrive - University of Strathclyde\General\00_Experimental Data\2025_02_05_Matt_SDH_polarity_Al_FMC\Scan 2 long\FMC_v7.mat")

# Create a params dict to fill and return:
params_al_sdh = sdh_fmc_param_set(fmc_2d=fmc_2d_SDH_raw_v,
                                  pitch_mm=0.2,
                                  t_min_us=-0.5,
                                  t_max_us=9.49,
                                  x_sdh_mm=9.75,
                                  z_sdh_mm=5,
                                  radius_sdh_mm=0.5,
                                  rise_time_us=0.06)
mss_b_scan_MWS_filtered_pm = convert_quartet_v_to_nm(
    params_al_sdh.get_mss_filtered(Wn_Hz=2*10**6)) * 1000

mss_b_scan_raw_pm = convert_quartet_v_to_nm(
    params_al_sdh.mss_b_scan) * 1000

# Additional SDH dataset parameters:
x_element_max_mm = np.max(params_al_sdh.x_elements_mm)
t_max_nmo_us = params_al_sdh.t_max_us - params_al_sdh.rise_time_us
t_min_nmo_us = params_al_sdh.t_min_us - params_al_sdh.rise_time_us
extent_SDH_dataset = [0, x_element_max_mm, t_max_nmo_us, t_min_nmo_us]


# 2mm Al NMO scan:
# Load the data:
params_al_nmo = get_params_aluminium_NMO_scan()
nmo_b_scan_raw_v = params_al_nmo.b_scan_array_2d_detrend
nmo_ds_detrend_pm = convert_quartet_v_to_nm(nmo_b_scan_raw_v) * 1000

# Generate axis vectors:
x_max_nmo_mm = params_al_nmo.x_max_mm
x_vector_nmo_mm = params_al_nmo.x_vector_mm
t_min_nmo_us = params_al_nmo.t_min_us
t_max_nmo_us = params_al_nmo.t_max_us
rise_time_nmo_us = params_al_nmo.rise_time_us
time_vector_nmo_us = params_al_nmo.time_vector_us
extent_NMO_dataset = [0, x_max_nmo_mm,
                      t_max_nmo_us-rise_time_nmo_us,
                      t_min_nmo_us-rise_time_nmo_us]


# Set plotting parameters:
c_map = cc.m_CET_D7
# Set displacement colormap limits:
v_min_SDH_pm = -30
v_max_SDH_pm = 30
v_min_NMO_pm = -100
v_max_NMO_pm = 100

# Plot each data array to make sure it is in the format we want:
# SDH dataset: Raw and MWS & filtered
fig_SDH_dataset, [ax_raw, ax_MWS_filtered] = plt.subplots(1, 2, layout='constrained')
im_SDH_raw = ax_raw.imshow(mss_b_scan_raw_pm,
                           extent=extent_SDH_dataset,
                           aspect='auto', cmap=c_map)
im_SDH_raw.set_clim(vmin=v_min_SDH_pm, vmax=v_max_SDH_pm)
ax_raw.set_title('Raw (pm)')
im_SDH_MWS_filt = ax_MWS_filtered.imshow(mss_b_scan_MWS_filtered_pm,
                                         extent=extent_SDH_dataset,
                                         aspect='auto', cmap=c_map)
im_SDH_MWS_filt.set_clim(vmin=v_min_SDH_pm, vmax=v_max_SDH_pm)
ax_MWS_filtered.set_title('MWS & Filtered (pm)')
fig_SDH_dataset.suptitle('SDH monostatic scan dataset')


# NMO dataset: Detrended only
fig_NMO_dataset, ax_nmo = plt.subplots(1, 1, layout='constrained')
im_NMO_raw = ax_nmo.imshow(nmo_ds_detrend_pm,
                           extent=extent_NMO_dataset,
                           aspect='auto', cmap=c_map)
im_NMO_raw.set_clim(vmin=v_min_NMO_pm, vmax=v_max_NMO_pm)
ax_nmo.set_title('Raw (pm)')


# Saving datasets:
# np.savez('Displacement_data',
#          experiment_1_raw_pm=mss_b_scan_raw_pm,
#          experiment_1_MWS_filtered_pm=mss_b_scan_MWS_filtered_pm,
#          experiment_2_raw_pm=nmo_ds_detrend_pm,
#          allow_pickle=False)


# Loading datasets:
datasets = np.load('Displacement_data.npz')

plt.imshow(datasets['experiment_2_raw_pm'])
