import colorcet as cc
import numpy as np
from scipy.signal import hilbert

from corevariables.modhfsexperiment import params_al_nmo
from functions.modnmocorrection import nmo_correct
from functions.modhwcorrect import cdp_head_wave_correct
from functions.modthetacrittrig import calculate_theta_crit_deg
from functions.modsumgather import sum_gather
from functions.modmyplotfuncs import plot_wiggles_DAS
from corevariables.modieeeplotting import (fontsize_labels_pt)
from functions.modbt0 import (calculate_b_mm_from_t_0_us)
from functions.modphasecompensation import (compute_C_G_SV_from_deg,
                                            compute_C_R_SV_SV_from_deg,
                                            compute_C_D_SV_from_deg,
                                            get_C_90_hybridS)


# IPython magic:
# %load_ext autoreload
# %autoreload 2

def phase_comp_plots_hfs(
    ax_colormap,
    ax_sum_waveforms,
    v_min_nm,
    v_max_nm
):

    # Load the data:
    b_scan_ds_nm = params_al_nmo.b_scan_array_2d_detrend_pm / 1000

    c_L_mpers = 6371  # Nemo
    c_T_mpers = 3077  # Nemo
    kappa_hfs = c_L_mpers / c_T_mpers

    # Generate axis vectors:
    x_max_mm = params_al_nmo.x_max_mm
    x_vector_mm = params_al_nmo.x_vector_mm
    time_vector_us = params_al_nmo.time_vector_us

    # Filter:
    # frequency_sampling_hertz = params_al_nmo.frequency_sampling_Hz
    # band_type = 'highpass'
    # Wn_MHz = 5
    # butter_order = 10
    # b_scan_ds_V = filter_fmc3d_butter(b_scan_ds_V_raw,
    #                                   frequency_sampling_hertz,
    #                                   butter_order,
    #                                   band_type,
    #                                   Wn_MHz)

    # Calculate specular ray angles:
    b_mm = 2.00
    angles_spec_rays_vector_deg = np.rad2deg(np.arctan2((x_vector_mm * 10**-3),
                                                        (2 * b_mm * 10**-3)))
    c_L_mpers = 6371  # Nemo
    c_T_mpers = 3077  # Nemo
    theta_crit_deg = calculate_theta_crit_deg(c_L_mpers, c_T_mpers)
    # Calculate the x values associated with 15 -degree intervals up to
    # 90 degrees:
    theta_max_deg = np.rad2deg(np.atan2(x_max_mm, 2 * b_mm))
    tick_angles_spaced_deg = np.linspace(0, 60, 5)
    tick_angles_deg = np.append(tick_angles_spaced_deg, theta_max_deg)
    ticks_mm = np.tan(np.deg2rad(tick_angles_deg)) * 2 * b_mm
    labels_angles = [f'{angle:.0f}' for angle in tick_angles_deg]

    # Generate NMO corrected extracts:
    t_0_half_width_us = 0.4
    n_t_0s = 200

    # Trial the same set of possible b-values for S-S as for H-S:
    t_0_TT_us = 1.28
    b_centre_shear_mm = calculate_b_mm_from_t_0_us(t_0_TT_us, c_T_mpers)
    b_half_width_shear_mm = calculate_b_mm_from_t_0_us(t_0_half_width_us,
                                                       c_T_mpers)
    b_min_shear_mm = b_centre_shear_mm - b_half_width_shear_mm
    b_max_shear_mm = b_centre_shear_mm + b_half_width_shear_mm

    # Make a vector of bs for plotting the wiggles:
    b_shear_vector_mm = np.linspace(b_min_shear_mm, b_max_shear_mm, n_t_0s)

    # Calculate phase compensation factors:
    C_G_HFS = compute_C_G_SV_from_deg(angles_spec_rays_vector_deg, kappa_hfs)
    C_R_HFS = compute_C_R_SV_SV_from_deg(angles_spec_rays_vector_deg,
                                         kappa_hfs)
    C_D_HFS = compute_C_D_SV_from_deg(angles_spec_rays_vector_deg, kappa_hfs)
    # Apply phase compensation factors:
    b_scan_ss_phase_comp_nm = np.real(hilbert(b_scan_ds_nm, axis=0)
                                      * C_G_HFS * C_R_HFS * C_D_HFS)

    # NMO correction:
    # T-T wavefront:
    ss_corr_phase_comp_nm, t_vector_SS_us = nmo_correct(b_scan_ss_phase_comp_nm,
                                                        time_vector_us,
                                                        t_0_TT_us,
                                                        t_0_half_width_us,
                                                        n_t_0s,
                                                        x_vector_mm,
                                                        c_T_mpers)

    # # Calculate phase compensation factors:
    # C_G_HFS = compute_C_G_SV_from_deg(angles_spec_rays_vector_deg, kappa_hfs)
    # C_R_HFS = compute_C_R_SV_SV_from_deg(angles_spec_rays_vector_deg,
    #                                      kappa_hfs)
    # C_D_HFS = compute_C_D_SV_from_deg(angles_spec_rays_vector_deg, kappa_hfs)
    # # Apply phase compensation factors:
    # ss_corr_phase_comp_nm = np.real(hilbert(b_scan_SS_corr_nm, axis=0)
    #                                 * C_G_HFS * C_R_HFS * C_D_HFS)

    # Sum delayed gathers over certain angular ranges:
    sum_ds_SS_phase_comp_all_nm = sum_gather(
        ss_corr_phase_comp_nm, angles_spec_rays_vector_deg)
    sum_ds_SS_phase_comp_0_crit_nm = sum_gather(ss_corr_phase_comp_nm,
                                                angles_spec_rays_vector_deg,
                                                theta_max_deg=theta_crit_deg)
    sum_ds_SS_phase_comp_48_64_nm = sum_gather(ss_corr_phase_comp_nm,
                                               angles_spec_rays_vector_deg,
                                               theta_start_deg=48,
                                               theta_max_deg=64)
    sum_ds_SS_phase_comp_sub_plus_super_nm = (sum_ds_SS_phase_comp_0_crit_nm +
                                              sum_ds_SS_phase_comp_48_64_nm)

    # Visualise the B-scan:
    # Get extreme yellow and blue colors:
    c_map = cc.m_CET_D7
    cmap_extreme_pos = c_map(1.0)
    cmap_extreme_neg = c_map(0.0)
    # Define global x_axis amplitude for wiggle plots:
    x_axis_amplitude = 27

    # Define plotting functions:

    def b_scan_axes_format(ax):
        # ax.set_ylabel(r'Time $t$ (μs)', fontsize=fontsize_labels_pt)
        ax.tick_params(top=True, labeltop=True,
                       bottom=False, labelbottom=False)
        ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
        ax.xaxis.set_label_position('top')

    def corr_plot_axes_format(ax):
        # Angle varies as the arctangent of the x_offset over b.
        # We can use non-periodic tick spacing to mark 10 degree angle
        # intervals.
        ax.set_xticks(ticks_mm, labels=labels_angles)
        # Label y-axis:
        ax.set_ylabel(r'$b$ (mm)', fontsize=fontsize_labels_pt)

    def theta_line(ax, angle_deg, label):
        # Plot a dashed axvline at the x value corresponding to the given
        # angle:
        x_theta_mm = np.tan(np.deg2rad(angle_deg)) * 2 * b_mm
        ax.axvline(x_theta_mm, color=(1, 1, 1), dashes=(4, 4), linewidth=0.7)
        t_lims = ax.get_ylim()
        t_text = t_lims[1] + ((t_lims[0] - t_lims[1]) * 0.96)
        ax.text(x_theta_mm, t_text, label, color=(1, 1, 1),
                fontsize=fontsize_labels_pt)

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

    # Plot NMO & phase-corrected T-T wavefront:
    extent_3 = [0, x_max_mm,
                b_max_shear_mm, b_min_shear_mm]
    im = ax_colormap.imshow(ss_corr_phase_comp_nm*1000, cmap=c_map,
                            extent=extent_3, aspect='auto')
    im.set_clim(vmin=v_min_nm*1000, vmax=v_max_nm*1000)
    b_scan_axes_format(ax_colormap)
    corr_plot_axes_format(ax_colormap)
    theta_line(ax_colormap, theta_crit_deg, r' $\theta$*')
    theta_line(ax_colormap, 48, ' 48')
    theta_line(ax_colormap, 64, ' 64')
    # ax_3.set_yticks([1, 1.5])
    axis_label(ax_colormap, '(b)')

    # Sum waveforms:
    plot_wiggles_DAS(ax_sum_waveforms,
                     [sum_ds_SS_phase_comp_0_crit_nm,
                      sum_ds_SS_phase_comp_48_64_nm,
                      sum_ds_SS_phase_comp_sub_plus_super_nm,
                      sum_ds_SS_phase_comp_all_nm],
                     b_shear_vector_mm,
                     [r'0-$\theta$*',
                      '48-64',
                      r'0-$\theta$*+48-64',
                      '0-68'],
                     x_axis_amplitude,
                     cmap_extreme_pos,
                     cmap_extreme_neg)
    ax_sum_waveforms.set_ylim(ax_colormap.get_ylim())
    axis_label(ax_sum_waveforms, '(d)')

    return im


def phase_comp_plots_hfs_with_hybrid(
    ax_colormap_SS,
    ax_colormap_hybrid_S,
    ax_sums_SS,
    ax_sums_hybrid_S,
    v_min_nm,
    v_max_nm
):

    # Load the data:
    b_scan_ds_nm = params_al_nmo.b_scan_array_2d_detrend_pm / 1000

    c_L_mpers = 6371  # Nemo
    c_T_mpers = 3077  # Nemo
    kappa_hfs = c_L_mpers / c_T_mpers

    # Generate axis vectors:
    x_max_mm = params_al_nmo.x_max_mm
    x_vector_mm = params_al_nmo.x_vector_mm
    time_vector_us = params_al_nmo.time_vector_us

    # Filter:
    # frequency_sampling_hertz = params_al_nmo.frequency_sampling_Hz
    # band_type = 'highpass'
    # Wn_MHz = 5
    # butter_order = 10
    # b_scan_ds_V = filter_fmc3d_butter(b_scan_ds_V_raw,
    #                                   frequency_sampling_hertz,
    #                                   butter_order,
    #                                   band_type,
    #                                   Wn_MHz)

    # Calculate specular ray angles:
    b_mm = 2.00
    angles_spec_rays_vector_deg = np.rad2deg(np.arctan2((x_vector_mm * 10**-3),
                                                        (2 * b_mm * 10**-3)))
    c_L_mpers = 6371  # Nemo
    c_T_mpers = 3077  # Nemo
    theta_crit_deg = calculate_theta_crit_deg(c_L_mpers, c_T_mpers)
    # Calculate the x values associated with 15 -degree intervals up to
    # 90 degrees:
    theta_max_deg = np.rad2deg(np.atan2(x_max_mm, 2 * b_mm))
    tick_angles_spaced_deg = np.linspace(0, 60, 5)
    tick_angles_deg = np.append(tick_angles_spaced_deg, theta_max_deg)
    ticks_mm = np.tan(np.deg2rad(tick_angles_deg)) * 2 * b_mm
    labels_angles = [f'{angle:.0f}' for angle in tick_angles_deg]

    # Generate NMO corrected extracts:
    t_0_half_width_us = 0.4
    n_t_0s = 200

    # Trial the same set of possible b-values for S-S as for H-S:
    t_0_TT_us = 1.28
    b_centre_shear_mm = calculate_b_mm_from_t_0_us(t_0_TT_us, c_T_mpers)
    b_half_width_shear_mm = calculate_b_mm_from_t_0_us(t_0_half_width_us,
                                                       c_T_mpers)
    b_min_shear_mm = b_centre_shear_mm - b_half_width_shear_mm
    b_max_shear_mm = b_centre_shear_mm + b_half_width_shear_mm

    # Make a vector of bs for plotting the wiggles:
    b_shear_vector_mm = np.linspace(b_min_shear_mm, b_max_shear_mm, n_t_0s)

    # Calculate phase compensation factors:
    C_G_HFS = compute_C_G_SV_from_deg(angles_spec_rays_vector_deg, kappa_hfs)
    C_R_HFS = compute_C_R_SV_SV_from_deg(angles_spec_rays_vector_deg,
                                         kappa_hfs)
    C_D_HFS = compute_C_D_SV_from_deg(angles_spec_rays_vector_deg, kappa_hfs)

    C_90_hybrid_S_HFS = get_C_90_hybridS(angles_spec_rays_vector_deg,
                                         theta_crit_deg)

    # Apply phase compensation factors:
    b_scan_ss_phase_comp_nm = np.real(hilbert(b_scan_ds_nm, axis=0)
                                      * C_G_HFS * C_R_HFS * C_D_HFS)

    b_scan_hybrid_S_phase_comp_nm = np.real(hilbert(b_scan_ds_nm, axis=0)
                                            * C_90_hybrid_S_HFS)

    # NMO correction:

    # T-T wavefront:
    corr_ss_phase_comp_nm, t_vector_SS_us = nmo_correct(
        b_scan_ss_phase_comp_nm,
        time_vector_us,
        t_0_TT_us,
        t_0_half_width_us,
        n_t_0s,
        x_vector_mm,
        c_T_mpers)

    # H-S correction:
    c_LSAW_mpers = 6000
    corr_HS_phase_comp_nm = cdp_head_wave_correct(
        b_scan_hybrid_S_phase_comp_nm,
        b_shear_vector_mm,
        x_vector_mm,
        time_vector_us,
        c_T_mpers,
        c_LSAW_mpers)

    # Build hybrid correction from S-S subcrit and H-S supercrit:
    # Calculate the indices in the corrected arrays associated with the
    # critical angle:
    (index_first_supercrit
     ) = np.argmax(angles_spec_rays_vector_deg > theta_crit_deg)
    # Slice corrected arrays:
    hybrid_corr_SS_part_nm = corr_ss_phase_comp_nm[:, 0:index_first_supercrit]
    hybrid_corr_HS_part_nm = corr_HS_phase_comp_nm[:, index_first_supercrit:]
    # Concatenate:
    corr_hybrid_s_phase_comp_nm = np.hstack(
        (hybrid_corr_SS_part_nm, hybrid_corr_HS_part_nm))

    # # Calculate phase compensation factors:
    # C_G_HFS = compute_C_G_SV_from_deg(angles_spec_rays_vector_deg, kappa_hfs)
    # C_R_HFS = compute_C_R_SV_SV_from_deg(angles_spec_rays_vector_deg,
    #                                      kappa_hfs)
    # C_D_HFS = compute_C_D_SV_from_deg(angles_spec_rays_vector_deg, kappa_hfs)
    # # Apply phase compensation factors:
    # ss_corr_phase_comp_nm = np.real(hilbert(b_scan_SS_corr_nm, axis=0)
    #                                 * C_G_HFS * C_R_HFS * C_D_HFS)

    # Sum delayed gathers over certain angular ranges:
    sum_ds_SS_phase_comp_all_nm = sum_gather(
        corr_ss_phase_comp_nm, angles_spec_rays_vector_deg)
    sum_ds_SS_phase_comp_0_crit_nm = sum_gather(corr_ss_phase_comp_nm,
                                                angles_spec_rays_vector_deg,
                                                theta_max_deg=theta_crit_deg)
    sum_ds_SS_phase_comp_48_64_nm = sum_gather(corr_ss_phase_comp_nm,
                                               angles_spec_rays_vector_deg,
                                               theta_start_deg=48,
                                               theta_max_deg=64)
    sum_ds_SS_phase_comp_sub_plus_super_nm = (sum_ds_SS_phase_comp_0_crit_nm +
                                              sum_ds_SS_phase_comp_48_64_nm)

    sum_ds_hybrid_S_phase_comp_all_nm = sum_gather(
        corr_hybrid_s_phase_comp_nm,
        angles_spec_rays_vector_deg
    )

    # Visualise the B-scan:
    # Get extreme yellow and blue colors:
    c_map = cc.m_CET_D7
    cmap_extreme_pos = c_map(1.0)
    cmap_extreme_neg = c_map(0.0)
    # Define global x_axis amplitude for wiggle plots:
    x_axis_amplitude = 27

    # Define plotting functions:

    def b_scan_axes_format(ax):
        # ax.set_ylabel(r'Time $t$ (μs)', fontsize=fontsize_labels_pt)
        ax.tick_params(top=True, labeltop=True,
                       bottom=False, labelbottom=False)
        ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
        ax.xaxis.set_label_position('top')

    def corr_plot_axes_format(ax):
        # Angle varies as the arctangent of the x_offset over b.
        # We can use non-periodic tick spacing to mark 10 degree angle
        # intervals.
        ax.set_xticks(ticks_mm, labels=labels_angles)
        # Label y-axis:
        ax.set_ylabel(r'$b$ (mm)', fontsize=fontsize_labels_pt)

    def theta_line(ax, angle_deg, label):
        # Plot a dashed axvline at the x value corresponding to the given
        # angle:
        x_theta_mm = np.tan(np.deg2rad(angle_deg)) * 2 * b_mm
        ax.axvline(x_theta_mm, color=(1, 1, 1), dashes=(4, 4), linewidth=0.7)
        t_lims = ax.get_ylim()
        t_text = t_lims[1] + ((t_lims[0] - t_lims[1]) * 0.96)
        ax.text(x_theta_mm, t_text, label, color=(1, 1, 1),
                fontsize=fontsize_labels_pt)

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

    # Plot NMO & phase-corrected T-T wavefront:
    extent_3 = [0, x_max_mm,
                b_max_shear_mm, b_min_shear_mm]
    im_SS = ax_colormap_SS.imshow(corr_ss_phase_comp_nm*1000, cmap=c_map,
                                  extent=extent_3, aspect='auto')
    im_SS.set_clim(vmin=v_min_nm*1000, vmax=v_max_nm*1000)
    b_scan_axes_format(ax_colormap_SS)
    corr_plot_axes_format(ax_colormap_SS)
    theta_line(ax_colormap_SS, theta_crit_deg, r' $\theta$*')
    theta_line(ax_colormap_SS, 48, ' 48')
    theta_line(ax_colormap_SS, 64, ' 64')
    # ax_3.set_yticks([1, 1.5])
    axis_label(ax_colormap_SS, '(b)')

    # SS sum waveforms:
    plot_wiggles_DAS(ax_sums_SS,
                     [sum_ds_SS_phase_comp_0_crit_nm,
                      sum_ds_SS_phase_comp_48_64_nm,
                      sum_ds_SS_phase_comp_sub_plus_super_nm,
                      sum_ds_SS_phase_comp_all_nm],
                     b_shear_vector_mm,
                     [r'0-$\theta$*',
                      '48-64',
                      r'0-$\theta$*+48-64',
                      '0-68'],
                     x_axis_amplitude,
                     cmap_extreme_pos,
                     cmap_extreme_neg)
    ax_sums_SS.set_ylim(ax_colormap_SS.get_ylim())
    axis_label(ax_sums_SS, '(e)')

    # Plot corrected and phase-compensated hybrid-s wavefront:
    im_hybrid_S = ax_colormap_hybrid_S.imshow(corr_hybrid_s_phase_comp_nm*1000,
                                              cmap=c_map,
                                              extent=extent_3,
                                              aspect='auto')
    im_hybrid_S.set_clim(vmin=v_min_nm*1000, vmax=v_max_nm*1000)
    b_scan_axes_format(ax_colormap_hybrid_S)
    corr_plot_axes_format(ax_colormap_hybrid_S)
    theta_line(ax_colormap_hybrid_S, theta_crit_deg, r' $\theta$*')
    # ax_3.set_yticks([1, 1.5])
    axis_label(ax_colormap_hybrid_S, '(c)')

    # Hybrid-S sum waveform:
    plot_wiggles_DAS(ax_sums_hybrid_S,
                     [hilbert(sum_ds_hybrid_S_phase_comp_all_nm)],
                     b_shear_vector_mm,
                     [r'0-68'],
                     x_axis_amplitude,
                     cmap_extreme_pos,
                     cmap_extreme_neg,
                     True,
                     True)
    ax_sums_hybrid_S.set_ylim(ax_colormap_SS.get_ylim())
    axis_label(ax_sums_hybrid_S, '(f)')

    return im_SS, im_hybrid_S
