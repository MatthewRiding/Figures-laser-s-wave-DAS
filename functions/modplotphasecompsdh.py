from scipy.signal import hilbert
import colorcet as cc
import numpy as np

from functions.modloadnmoperiodicampsfrommatfile import load_nmo_periodic_amps_from_mat_file
from classdefs.modparamsets import material_param_set, sdh_fmc_param_set
from functions.modsdhcorrect import sdh_correct_t_0_based
from functions.modthetacrittrig import calculate_theta_crit_deg
from functions.modsumgather import sum_gather_pos_and_neg_angles
from functions.modmyplotfuncs import (plot_wiggles_DAS,
                                      plot_wiggles_DAS_central_20_percent)
from corevariables.modieeeplotting import (fontsize_labels_pt)
from functions.modconvertvtonm import convert_quartet_v_to_nm
from functions.modphasecompensation import (compute_C_G_SV_from_deg,
                                            compute_C_D_SV_from_deg)
from functions.modcalculationsfortable import get_table_row_sdh


# IPython magic:
# %load_ext autoreload
# %autoreload 2

def phase_comp_plots_sdh(ax_colormap, ax_sum_waveforms, v_min_nm, v_max_nm):

    # Load the data:
    fmc_2d_raw_v = load_nmo_periodic_amps_from_mat_file(
        r"C:\Users\mattr\OneDrive - University of Strathclyde\General\00_Experimental Data\2025_02_05_Matt_SDH_polarity_Al_FMC\Scan 2 long\FMC_v7.mat")

    # Create a params dict to fill and return:
    params_al_sdh = sdh_fmc_param_set(fmc_2d=fmc_2d_raw_v,
                                      pitch_mm=0.2,
                                      t_min_us=-0.5,
                                      t_max_us=9.49,
                                      x_sdh_mm=9.75,
                                      z_sdh_mm=5,
                                      radius_sdh_mm=0.5,
                                      rise_time_us=0.06)
    mss_b_scan_nm = convert_quartet_v_to_nm(
        params_al_sdh.get_mss_filtered(Wn_Hz=2*10**6))

    material_al = material_param_set(c_L_mpers=6475,
                                     c_T_mpers=3170)

    # Axis parameters:
    # I'm going to plot the MSS B-scan:
    x_element_max_mm = np.max(params_al_sdh.x_elements_mm)

    # Calculate angles of rays from (x_element, 0) passing through the
    # centre of the SDH (x_sdh, z_sdh):
    x_diff_mm = params_al_sdh.x_elements_mm - params_al_sdh.x_sdh_mm
    theta_array_deg = np.rad2deg(np.atan2(x_diff_mm, params_al_sdh.z_sdh_mm))
    theta_crit_deg = calculate_theta_crit_deg(material_al.c_L_mpers,
                                              material_al.c_T_mpers)
    # Calculate the x values associated with 15 -degree intervals up to 90
    # degrees so that you can plot ticks on the x-axis for the corrected
    # B-scans:
    theta_max_deg = np.max(theta_array_deg)
    theta_min_deg = np.min(theta_array_deg)
    tick_angles_pos_deg = np.linspace(0, 60, 5)
    tick_angles_neg_deg = -1 * np.flip(np.append(tick_angles_pos_deg[1:],
                                                 np.abs(theta_min_deg)))
    tick_angles_deg = np.append(np.append(tick_angles_neg_deg,
                                          tick_angles_pos_deg), theta_max_deg)
    ticks_mm = params_al_sdh.x_sdh_mm + (np.tan(np.deg2rad(tick_angles_deg)) *
                                         params_al_sdh.z_sdh_mm)
    labels_angles = [f'{angle:.0f}' for angle in np.abs(tick_angles_deg)]

    # Calculate phase compensation factors:
    C_G_SDH = compute_C_G_SV_from_deg(theta_array_deg, material_al.kappa)
    C_D_SDH = compute_C_D_SV_from_deg(theta_array_deg, material_al.kappa)
    # Apply phase compensation factors:
    mss_ss_phase_comp_nm = np.real(hilbert(mss_b_scan_nm, axis=0)
                                   * C_G_SDH * C_D_SDH)

    # Apply move-out corrections:
    # Set half width to use in correction functions:
    t_0_half_width_us = 0.6
    n_t_0s = 300

    # S-S wavefront:
    t_0_SS_sdh_us = 2.75
    (mss_ss_corr_phase_comp_nm, t_0_vector_ss_us) = sdh_correct_t_0_based(
        mss_ss_phase_comp_nm,
        t_0_SS_sdh_us,
        t_0_half_width_us,
        n_t_0s,
        params_al_sdh.x_elements_mm,
        params_al_sdh.time_vector_us,
        material_al.c_T_mpers,
        params_al_sdh.x_sdh_mm,
        params_al_sdh.radius_sdh_mm)
    z_apex_vector_ss_mm = (((t_0_vector_ss_us * 10**-6) / 2)
                           * material_al.c_T_mpers) / (10**-3)

    # # Calculate phase compensation factors:
    # C_G_SDH = compute_C_G_SV_from_deg(theta_array_deg, material_al.kappa)
    # C_D_SDH = compute_C_D_SV_from_deg(theta_array_deg, material_al.kappa)
    # # Apply phase compensation factors:
    # mss_ss_corr_phase_comp_nm = np.real(
    #     hilbert(mss_ss_corr_nm, axis=0) * C_G_SDH * C_D_SDH)

    # Sum delayed gathers over certain angular ranges:
    # SV-SV direct after phase compensation:
    # Sub-critical:
    sum_SS_phase_comp_0_crit_nm = sum_gather_pos_and_neg_angles(
        mss_ss_corr_phase_comp_nm,
        theta_array_deg,
        0, theta_crit_deg)
    # Supercritical:
    sum_SS_phase_comp_50_60_nm = sum_gather_pos_and_neg_angles(
        mss_ss_corr_phase_comp_nm,
        theta_array_deg,
        50, 60)
    # Subcrit direct + supercrit direct:
    sum_SS_phase_comp_sub_plus_super_nm = (sum_SS_phase_comp_0_crit_nm +
                                           sum_SS_phase_comp_50_60_nm)
    # All:
    sum_SS_phase_comp_0_60_nm = sum_gather_pos_and_neg_angles(
        mss_ss_corr_phase_comp_nm,
        theta_array_deg,
        0, 60)

    # Get row of table:
    theta_sep_deg = 50
    table_row_SDH_SS_phase_comp = get_table_row_sdh(
        mss_ss_corr_phase_comp_nm,
        theta_array_deg,
        theta_crit_deg,
        theta_sep_deg,
        60,
        30
    )

    # Get extreme yellow and blue colors:
    c_map = cc.m_CET_D7
    cmap_extreme_pos = c_map(1.0)
    cmap_extreme_neg = c_map(0.0)
    # Define global x_axis amplitude for wiggle plots:
    x_axis_amplitude = 0.43 * 10

    # Define plotting functions:

    def b_scan_axes_format(ax):
        # ax.set_ylabel(r'Time $t$ (μs)', fontsize=fontsize_labels_pt)
        ax.tick_params(top=True, labeltop=True,
                       bottom=False, labelbottom=False)
        ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
        ax.xaxis.set_label_position('top')

    def b_scan_x_axis_angle(ax):
        # Angle varies as the arctangent of the x_offset over b.
        # We can use non-periodic tick spacing to mark 10 degree
        # angle intervals.
        ax.set_xticks(ticks_mm, labels=labels_angles)

    def theta_lines(ax, angle_deg, label=None):
        # Plot dashed axvline at the x values corresponding to the
        # given angle:
        x_diff_theta_mm = np.tan(np.deg2rad(
            angle_deg)) * params_al_sdh.z_sdh_mm
        x_theta_pos_mm = params_al_sdh.x_sdh_mm + x_diff_theta_mm
        x_theta_neg_mm = params_al_sdh.x_sdh_mm - x_diff_theta_mm
        ax.axvline(x_theta_pos_mm, color=(1, 1, 1),
                   dashes=(4, 4), linewidth=0.7)
        ax.axvline(x_theta_neg_mm, color=(1, 1, 1),
                   dashes=(4, 4), linewidth=0.7)
        if label:
            y_lims = ax.get_ylim()
            y_text = y_lims[1] + ((y_lims[0] - y_lims[1]) * 0.93)
            ax.text(x_theta_pos_mm, y_text, label, color=(1, 1, 1),
                    fontsize=fontsize_labels_pt)
            ax.text(x_theta_neg_mm, y_text, label, color=(1, 1, 1),
                    fontsize=fontsize_labels_pt)

    def x_line(ax, x_mm, label_string=None):
        # Plot a dashed axvline at the x value given:
        ax.axvline(x_mm, color=(1, 1, 1), dashes=(4, 4), linewidth=0.7)
        if label_string:
            # Label the line with text:
            t_lims = ax.get_ylim()
            t_text = t_lims[1] + ((t_lims[0] - t_lims[1]) * 0.78)
            ax.text(x_mm, t_text, label_string, color=(1, 1, 1),
                    fontsize=fontsize_labels_pt, ha='left')

    def axis_label(ax, label):
        ax.set_title(label, fontsize=fontsize_labels_pt, y=0, pad=-9)

    def wave_set_label(ax, label):
        # Place a white text box with a black line border in the top
        # left corner of ax:
        props = dict(boxstyle='square', facecolor='w', edgecolor='k',
                     lw=0.7)
        ax.text(0.018, 0.96, label, transform=ax.transAxes,
                fontsize=fontsize_labels_pt,
                verticalalignment='top', bbox=props)

    # Plot moveout-corrected, phase-compensated S-S wavefront:
    extent_1 = [0, x_element_max_mm,
                np.max(z_apex_vector_ss_mm), np.min(z_apex_vector_ss_mm)]
    # Plot colorma in picometres:
    im = ax_colormap.imshow(mss_ss_corr_phase_comp_nm*1000, cmap=c_map,
                            extent=extent_1, aspect='auto')
    im.set_clim(vmin=v_min_nm*1000, vmax=v_max_nm*1000)
    b_scan_axes_format(ax_colormap)
    b_scan_x_axis_angle(ax_colormap)
    ax_colormap.set_xlabel(r'Ray angle $\theta$ ' r'(°)',
                           fontsize=fontsize_labels_pt)
    # Add dashed line marking x_SDH:
    x_line(ax_colormap, params_al_sdh.x_sdh_mm, r' $x_{SDH}$')
    theta_lines(ax_colormap, theta_crit_deg, r' $\theta$*')
    theta_lines(ax_colormap, 60, ' 60')
    theta_lines(ax_colormap, 50, ' 50')
    ax_colormap.set_yticks([4, 5])
    axis_label(ax_colormap, '(a)')
    # Add an axis label: z_apex (mm)
    ax_colormap.set_ylabel(r'$z_{\mathrm{apex}}$ (mm)',
                           fontsize=fontsize_labels_pt)

    # Summed waveform plot:
    plot_wiggles_DAS_central_20_percent(ax_sum_waveforms,
                                        [hilbert(sum_SS_phase_comp_0_crit_nm),
                                         hilbert(sum_SS_phase_comp_50_60_nm),
                                         hilbert(sum_SS_phase_comp_sub_plus_super_nm),
                                         hilbert(sum_SS_phase_comp_0_60_nm)],
                                        z_apex_vector_ss_mm,
                                        [r'0-$\theta$*',
                                            r'50-60',
                                            r'0-$\theta$*+50-60',
                                            r'0-60'],
                                        x_axis_amplitude,
                                        cmap_extreme_pos, cmap_extreme_neg,
                                        True, True,
                                        ['left', 'left', 'left', -0.25])
    ax_sum_waveforms.set_ylim(np.max(z_apex_vector_ss_mm),
                              np.min(z_apex_vector_ss_mm))
    axis_label(ax_sum_waveforms, '(c)')

    return im
