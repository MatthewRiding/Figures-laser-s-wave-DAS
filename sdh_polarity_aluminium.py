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

from classdefs.modparamsets import material_param_set, sdh_fmc_param_set


def get_material_params_aluminium():
    # c_L_mpers = 6375  # Dakota NDT
    # c_T_mpers = 3130  # Dakota NDT

    c_L_mpers = 6373  # Nemo
    c_T_mpers = 3048  # Nemo

    params = material_param_set(c_L_mpers, c_T_mpers)
    return params


def get_params_aluminium_SDH_scan():
    # Data from Aluminium SDH experiment (2025_02_05):

    # Load the data:
    fmc_2d_raw = load_nmo_periodic_amps_from_mat_file(
        r"C:\Users\mattr\OneDrive - University of Strathclyde\General\00_Experimental Data\2025_02_05_Matt_SDH_polarity\Scan 2 long\FMC.mat")

    # Create a params dict to fill and return:
    params = sdh_fmc_param_set(fmc_2d=fmc_2d_raw,
                               pitch_mm=0.2,
                               t_min_us=-0.5,
                               t_max_us=9.49,
                               x_sdh_mm=9.8,
                               z_sdh_mm=5,
                               radius_sdh_mm=0.5,
                               rise_time_us=0.06)
    return params


def aluminium_sdh_cmp(params_sdh):
    # Extract different gathers:
    # Common mid-point:
    cmp_gen_high = extract_cmp_from_fmc2d(params_sdh.fmc_2d_detrend, 'gen_high')
    cmp_gen_low = extract_cmp_from_fmc2d(params_sdh.fmc_2d_detrend, 'gen_low')

    # Make CMP figure:
    fig, (ax_1, ax_2) = plt.subplots(1, 2)

    im_1 = ax_1.imshow(cmp_gen_high*1000, cmap=params_sdh.c_map, aspect='auto')
    im_1.set_clim(vmin=params_sdh.c_min_mV, vmax=params_sdh.c_max_mV)

    im_2 = ax_2.imshow(cmp_gen_low*1000, cmap=params_sdh.c_map, aspect='auto')
    im_2.set_clim(vmin=params_sdh.c_min_mV, vmax=params_sdh.c_max_mV)


def aluminium_sdh_mss_TT_corr(params_sdh, params_al, t_0_half_width_us, n_t_0s,
                              t_0_TT_us):

    mss_filtered = params_sdh.get_mss_filtered(Wn_MHz=2)

    # Correct for move-out:
    (mss_corr, t_0_vector_us) = sdh_correct_t_0_based(mss_filtered,
                                            t_0_TT_us,
                                            t_0_half_width_us,
                                            n_t_0s,
                                            params_sdh.x_elements_mm,
                                            params_sdh.time_vector_us,
                                            params_al.c_T_mpers,
                                            params_sdh.x_sdh_mm,
                                            params_sdh.radius_sdh_mm)

    return mss_filtered, mss_corr, t_0_vector_us


if __name__ == '__main__':
    # Import parameters:
    params_sdh = get_params_aluminium_SDH_scan()
    params_al = get_material_params_aluminium()

    # Apply move-out correction:
    t_0_half_width_us = 0.6
    n_t_0s = 300
    t_0_TT_us = 2.8
    mss_filtered, mss_corr = aluminium_sdh_mss_TT_corr(params_sdh, params_al,
                                                       t_0_half_width_us,
                                                       n_t_0s,
                                                       t_0_TT_us)
    t_0_min_sdh_us = t_0_TT_us - t_0_half_width_us
    t_0_max_sdh_us = t_0_TT_us + t_0_half_width_us

    # Calculate x positions corresponding to angles of interest:
    x_crit_mm = params_sdh.x_sdh_mm + (params_sdh.z_sdh_mm *
                                       params_al.tan_theta_crit)
    x_45_mm = params_sdh.x_sdh_mm + 5

    # Plotting parameters:
    c_map = cc.m_CET_D7

    # Make MSS figure:
    fig, (ax_3, ax_4) = plt.subplots(2, 1)

    im_3 = ax_3.imshow(mss_filtered*1000, cmap=c_map, aspect='auto',
                       extent=(0, params_sdh.aperture_mm,
                               params_sdh.t_max_us, params_sdh.t_min_us))
    ax_3.set_xlabel('x (mm)')
    ax_3.set_ylabel(r'Time ($\mu$s)')
    im_3.set_clim(vmin=-10, vmax=10)

    im_4 = ax_4.imshow(mss_corr*1000, cmap=c_map, aspect='auto',
                       extent=(0, params_sdh.aperture_mm,
                               t_0_max_sdh_us,
                               t_0_min_sdh_us))
    ax_4.set_xlabel('x (mm)')
    ax_4.set_ylabel(r'Time ($\mu$s)')
    im_4.set_clim(vmin=-5, vmax=5)
    ax_4.axvline(x_crit_mm, ls='dashed', color='w')
    ax_4.axvline(x_45_mm, ls='dashed', color='k')
    # ax_4.set_xlim((x_sdh_mm, 20))

    plt.show()
