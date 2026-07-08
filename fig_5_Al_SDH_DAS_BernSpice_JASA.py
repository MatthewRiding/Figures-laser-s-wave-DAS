# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import hilbert
import colorcet as cc
import numpy as np

from corevariables.modsdhexperiment import (
    params_al_sdh,
    material_al_sdh
)
from corevariables.modjasa import lw_boundaries
from functions.modsdhcorrect import (
    sdh_correct_HS_z_apex_based,
    sdh_correct_XX_using_z_apex_vector,
    sdh_correct_HS_bernspice)
from functions.modsumgather import sum_gather_pos_and_neg_angles
from functions.modmyplotfuncs import (plot_wiggles_DAS_central_20_percent)
from corevariables.modjasa import (fontsize_labels_pt,
                                   width_fig_two_column_jasa_in)
from functions.modmyplotfuncs import set_mpl_fonts_Times_New_Roman
from functions.modconvertvtonm import convert_quartet_v_to_nm
from functions.modraydiagrammss import plot_ray_diagram_sdh_mss
from bernspice_sdh import get_bernspice_a_scans_sdh
from functions.modcalculationsfortable import get_table_row_sdh

# %%

# IPython magic:
# %load_ext autoreload
# %autoreload 2


# Experiment data imported from corevariables.modsdhexperiment.
# Most data contained within the object called 'params_al_sdh'.

mss_b_scan_nm = convert_quartet_v_to_nm(
    params_al_sdh.get_mss_filtered(Wn_Hz=2*10**6))

# Axis parameters:
# I'm going to plot the MSS B-scan:
x_element_max_mm = np.max(params_al_sdh.x_elements_mm)
t_max_us = params_al_sdh.t_max_us - params_al_sdh.rise_time_us
t_min_us = params_al_sdh.t_min_us - params_al_sdh.rise_time_us

# Get vector of angles (theta) of rays from (x_element, 0) passing through the
# centre of the SDH (x_sdh, z_sdh):
theta_array_deg = params_al_sdh.angles_gen_sdh_deg
# Get critical angle:
theta_crit_deg = material_al_sdh.theta_crit_deg
# Get surface offset from SDH that gives critical ray angle:
x_crit_mm = material_al_sdh.tan_theta_crit * params_al_sdh.z_sdh_mm
# Calculate the x values associated with 15 -degree intervals up to 90 degrees
# so that you can plot ticks on the x-axis for the corrected B-scans:
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

# Apply move-out corrections:
# Specify 'imaging vector'- a vector of z_apex values to
# perform moveout corrections for (z_apex is the  depth of
# the apex of the SDH):
z_apex_start_mm = 3.25
z_apex_stop_mm = 5.25
n_z_apexs = 300
z_apex_vector_mm = np.linspace(z_apex_start_mm, z_apex_stop_mm, n_z_apexs)
# Set half width to use in correction functions:
t_0_half_width_us = 0.6
n_t_0s = 300

# L-L wavefront:
(corr_LL_exp_nm
 ) = sdh_correct_XX_using_z_apex_vector(mss_b_scan_nm,
                                        z_apex_vector_mm,
                                        params_al_sdh.time_vector_us,
                                        material_al_sdh.c_L_mpers)

# S-S wavefront:
(corr_SS_exp_nm
 ) = sdh_correct_XX_using_z_apex_vector(mss_b_scan_nm,
                                        z_apex_vector_mm,
                                        params_al_sdh.time_vector_us,
                                        material_al_sdh.c_T_mpers)

# H-S wavefront:
# Use same y-axis range as S-S so that they match horizontally and
# the hybrid-corrected wavefront can be built easily:
c_LSW_mpers = 6000
corr_HS_exp_nm = sdh_correct_HS_z_apex_based(mss_b_scan_nm,
                                             z_apex_vector_mm,
                                             params_al_sdh.time_vector_us,
                                             c_LSW_mpers)
# np.save(r'saved_numpy_variables\Al_MSS_HS_corr_nm', corr_HS_exp_nm)

# Create the hybrid-corrected wavefront by using the S-S direct correction up
# until theta_crit, then using the H-T correction from theta-crit upwards.
# Calculate the indices in the corrected arrays associated with the critical
# angle:
index_first_subcrit = np.argmax((theta_array_deg + theta_crit_deg) > 0)
index_first_supercrit_right = np.argmax((theta_array_deg - theta_crit_deg) > 0)
# Slice corrected experimental wavefronts:
corr_hybrid_S_exp_left_nm = corr_HS_exp_nm[:, 0:index_first_subcrit]
(corr_hybrid_S_middle_exp_nm
 ) = (corr_SS_exp_nm)[:, index_first_subcrit:index_first_supercrit_right]
corr_hybrid_S_exp_right_nm = corr_HS_exp_nm[:, index_first_supercrit_right:]
corr_hybrid_S_exp_nm = np.hstack((corr_hybrid_S_exp_left_nm,
                                  corr_hybrid_S_middle_exp_nm,
                                  corr_hybrid_S_exp_right_nm))

# Sum delayed experimental gathers over certain angular ranges:
# L-L:
sum_LL_all_exp_nm = np.sum((corr_LL_exp_nm), axis=1)

# S-S:
# Sub-critical:
sum_SS_0_crit_exp_nm = sum_gather_pos_and_neg_angles((corr_SS_exp_nm),
                                                     theta_array_deg,
                                                     0, theta_crit_deg)
sum_SS_50_60_exp_nm = sum_gather_pos_and_neg_angles((corr_SS_exp_nm),
                                                    theta_array_deg,
                                                    50, 60)
sum_SS_0_crit_plus_50_60_exp_nm = sum_SS_0_crit_exp_nm + sum_SS_50_60_exp_nm
sum_SS_0_60_exp_nm = sum_gather_pos_and_neg_angles((corr_SS_exp_nm),
                                                   theta_array_deg,
                                                   0, 60)

# H-S & S-H
sum_HS_crit_60_exp_nm = sum_gather_pos_and_neg_angles(corr_HS_exp_nm,
                                                      theta_array_deg,
                                                      theta_crit_deg, 60)

# Hybrid:
sum_hybrid_S_0_60_exp_nm = sum_gather_pos_and_neg_angles(corr_hybrid_S_exp_nm,
                                                         theta_array_deg,
                                                         0, 60)


# %% Run the Bernstein and Spicer model:
# Compute theoretical MSS a-scans:

# Generate synthetic A-scans:
n_times_theoretical = 4000
time_vector_theoretical_us = np.linspace(t_min_us,
                                         t_max_us,
                                         n_times_theoretical)
# A-scans returned in rows of ndarray.
(u_z_ascans_sdh_bernspice_m
 ) = get_bernspice_a_scans_sdh(time_vector_theoretical_us * 10**-6)

# Get NMO-corrected extracts:
# L-L wavefront:
corr_LL_theo = sdh_correct_XX_using_z_apex_vector(
    u_z_ascans_sdh_bernspice_m.T,
    z_apex_vector_mm,
    time_vector_theoretical_us,
    material_al_sdh.c_L_mpers
)

# T-T wavefront:
corr_SS_theo = sdh_correct_XX_using_z_apex_vector(
    u_z_ascans_sdh_bernspice_m.T,
    z_apex_vector_mm,
    time_vector_theoretical_us,
    material_al_sdh.c_T_mpers
)

# H-T wavefront:
corr_HS_theo = sdh_correct_HS_bernspice(
    u_z_ascans_sdh_bernspice_m,
    z_apex_vector_mm,
    time_vector_theoretical_us)

# Create the hybrid-corrected wavefront by using the S-S direct correction up
# until theta_crit, then using the H-T correction from theta-crit upwards.
# Calculate the indices in the corrected arrays associated with the critical
# angle:
# Slice theoretical corrected arrays:
corr_hybrid_S_left_theo = corr_HS_theo[:, 0:index_first_subcrit]
corr_hybrid_S_middle_theo = (corr_SS_theo)[
    :, index_first_subcrit:index_first_supercrit_right]
corr_hybrid_S_right_theo = corr_HS_theo[:, index_first_supercrit_right:]
corr_hybrid_S_theo = np.hstack((corr_hybrid_S_left_theo,
                                corr_hybrid_S_middle_theo,
                                corr_hybrid_S_right_theo))

# Sum the corrected theoretical gathers over certain angular ranges:
# L-L:
sum_LL_all_theo = np.sum(corr_LL_theo, axis=1)

# S-S:
# Sub-critical:
sum_SS_0_crit_theo = sum_gather_pos_and_neg_angles(corr_SS_theo,
                                                   theta_array_deg,
                                                   0, theta_crit_deg)
sum_SS_50_60_theo = sum_gather_pos_and_neg_angles(corr_SS_theo,
                                                  theta_array_deg,
                                                  50, 60)
sum_SS_0_crit_plus_50_60_theo = sum_SS_0_crit_theo + sum_SS_50_60_theo
sum_SS_0_60_theo = sum_gather_pos_and_neg_angles(corr_SS_theo,
                                                 theta_array_deg,
                                                 0, 60)

# H-S & S-H
sum_HS_crit_60_theo = sum_gather_pos_and_neg_angles(corr_HS_theo,
                                                    theta_array_deg,
                                                    theta_crit_deg, 60)

# Hybrid:
sum_hybrid_S_0_60_theo = sum_gather_pos_and_neg_angles(corr_hybrid_S_theo,
                                                       theta_array_deg,
                                                       0, 60)

# Normalisation of theoretical sum waveforms:

# The theoretical sum waveforms have arbitrary absolute amplitude.
# To ease comparison against the experimental waveforms,
# we will set the maximum absolute value of the theoretical
# L-L sum waveform to 4.00 (the maximum absolute value of the
# experimental L-L sum waveform), and then normalise all of the
# other theoretical sum waveforms relative to the L-L.

# Scale the L-L theo sum waveform so that it has the same
# max abs value as the exp L-L sum waveform:
scale_factor_theo = (np.max(np.abs(hilbert(sum_LL_all_exp_nm))) /
                     np.max(np.abs(hilbert(sum_LL_all_theo)))
                     )

sum_LL_all_theo_norm = (sum_LL_all_theo * scale_factor_theo)

# Apply the same scale factor to all the other theoretical sum
# waveforms:
# S-S:
sum_SS_0_crit_theo_norm = sum_SS_0_crit_theo * scale_factor_theo
sum_SS_50_60_theo_norm = sum_SS_50_60_theo * scale_factor_theo
sum_SS_0_crit_plus_50_60_theo_norm = (sum_SS_0_crit_plus_50_60_theo *
                                      scale_factor_theo)
sum_SS_0_60_theo_norm = sum_SS_0_60_theo * scale_factor_theo

# Head-S:
sum_HS_crit_60_theo_norm = sum_HS_crit_60_theo * scale_factor_theo

# Hybrid-S:
sum_hybrid_S_0_60_theo_norm = sum_hybrid_S_0_60_theo * scale_factor_theo

# %% Get rows of the results table:

theta_sep_deg = 50

# Experimental:
row_ll_exp = get_table_row_sdh(corr_LL_exp_nm,
                               theta_array_deg,
                               theta_crit_deg,
                               theta_sep_deg,
                               90,
                               40)
row_ss_exp = get_table_row_sdh(corr_SS_exp_nm,
                               theta_array_deg,
                               theta_crit_deg,
                               theta_sep_deg,
                               60,
                               40)
row_hybrid_s_exp = get_table_row_sdh(corr_hybrid_S_exp_nm,
                                     theta_array_deg,
                                     theta_crit_deg,
                                     theta_sep_deg,
                                     60,
                                     40)

# Modelled:
row_ll_theo = get_table_row_sdh(corr_LL_theo * scale_factor_theo,
                                theta_array_deg,
                                theta_crit_deg,
                                theta_sep_deg,
                                90,
                                40)
row_ss_theo = get_table_row_sdh(corr_SS_theo * scale_factor_theo,
                                theta_array_deg,
                                theta_crit_deg,
                                theta_sep_deg,
                                60,
                                40)
row_hybrid_s_theo = get_table_row_sdh(corr_hybrid_S_theo * scale_factor_theo,
                                      theta_array_deg,
                                      theta_crit_deg,
                                      theta_sep_deg,
                                      60,
                                      40)

# Stack rows into arrays that you can view in the debugger:
table_array_exp = np.vstack((row_ll_exp,
                             row_ss_exp,
                             row_hybrid_s_exp))
table_array_theo = np.vstack((row_ll_theo,
                              row_ss_theo,
                              row_hybrid_s_theo))


# %% Define plotting parameters & functions:

# Figure dimensions:
fig_width_inches = width_fig_two_column_jasa_in
fig_height_inches = 4/3 * 6 * 0.9
fig_dpi = 200
set_mpl_fonts_Times_New_Roman()

# Set experimental displacement colormap limits:
v_min_exp_nm = -0.03
v_max_exp_nm = 0.03
cbar_ticks_exp_pm = [-30, 0, 30]
# Theoretical displacement colormap limits:
v_min_theo_m = -5 * 10**-9
v_max_theo_m = - v_min_theo_m
cbar_ticks_theo = [v_min_theo_m,
                   0,
                   v_max_theo_m]
# Get extreme yellow and blue colors:
c_map = cc.m_CET_D7
cmap_extreme_pos = c_map(1.0)
cmap_extreme_neg = c_map(0.0)
# Define global x_axis amplitude for wiggle plots:
x_axis_amplitude_wiggle_plots = 5.7
# Extent for imshow of B-scans:
extent_b_scan = [0, x_element_max_mm, t_max_us, t_min_us]
# Extent for corr array imshows:
extent_corr = [0, x_element_max_mm,
               np.max(z_apex_vector_mm), np.min(z_apex_vector_mm)]

# Define plotting functions:


def image_axes_format(ax):
    # ax.set_ylabel(r'Time $t$ (μs)', fontsize=fontsize_labels_pt)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax.xaxis.set_label_position('top')


def corr_plot_ax_format(ax):
    image_axes_format(ax)
    # Angle varies as the arctangent of the x_offset over b.
    # We can use non-periodic tick spacing to mark 10 degree angle intervals.
    ax.set_xticks(ticks_mm, labels=labels_angles)
    # Label the y-axis: z_apex (mm)
    ax.set_ylabel(r'$z_{\mathrm{apex}}$ (mm)', fontsize=fontsize_labels_pt)
    # Set the y-ticks in mm:
    ax.set_yticks([4, 5])


def theta_lines(ax, angle_deg, label=None):
    # Plot dashed axvlines at the two x values corresponding to +/- the given
    # angle (either side of the SDH):
    x_diff_theta_mm = np.tan(np.deg2rad(angle_deg)) * params_al_sdh.z_sdh_mm
    x_theta_pos_mm = params_al_sdh.x_sdh_mm + x_diff_theta_mm
    x_theta_neg_mm = params_al_sdh.x_sdh_mm - x_diff_theta_mm
    ax.axvline(x_theta_pos_mm, color=(1, 1, 1), dashes=(4, 4), linewidth=0.7)
    ax.axvline(x_theta_neg_mm, color=(1, 1, 1), dashes=(4, 4), linewidth=0.7)
    if label:
        y_lims = ax.get_ylim()
        y_text = y_lims[1] + ((y_lims[0] - y_lims[1]) * 0.93)
        ax.text(x_theta_pos_mm, y_text, label, color=(1, 1, 1),
                fontsize=fontsize_labels_pt)
        ax.text(x_theta_neg_mm, y_text, label, color=(1, 1, 1),
                fontsize=fontsize_labels_pt)


def x_line(ax, x_mm, label_string=None, y_factor=0.93):
    # Plot a dashed axvline at the x value given:
    ax.axvline(x_mm, color=(1, 1, 1), dashes=(4, 4), linewidth=0.7)
    if label_string:
        # Label the line with text:
        t_lims = ax.get_ylim()
        t_text = t_lims[1] + ((t_lims[0] - t_lims[1]) * y_factor)
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


def plot_b_scan(
    ax,
    fig_top,
    displacements,
    v_min,
    v_max,
    c_bar_ticks,
    string_c_bar_label,
    cbar_tick_labels=None,
    label_x=False
):
    # Plot colormap in picometres:
    im_b_scan = ax.imshow(displacements,
                          cmap=c_map,
                          extent=extent_b_scan,
                          aspect='auto')
    im_b_scan.set_clim(vmin=v_min, vmax=v_max)
    image_axes_format(ax)
    # Set y-axis (time) limits in microseconds:
    ax.set_ylim(7.5, 0)

    # Label axes:
    if label_x:
        ax.set_xlabel(r'Monostatic position $x$ (mm)',
                      fontsize=fontsize_labels_pt)
    else:
        ax.set_xticks(ax.get_xticks(), [])
    ax.set_ylabel(r'Time $t$ (μs)', fontsize=fontsize_labels_pt)

    # Add dashed line marking x_SDH:
    x_line(ax, params_al_sdh.x_sdh_mm,
           r' $x_{\mathrm{SDH}}$',
           y_factor=0.7)

    # Add a white rectangle patch to backdrop the colorbar:
    width_rect = 0.6
    height_rect = 0.19
    x_rect_start = 0.5 - (width_rect / 2)
    y_rect_start = 0.025
    patch_rect = Rectangle((x_rect_start, y_rect_start),
                           width_rect,
                           height_rect,
                           transform=ax.transAxes,
                           lw=lw_boundaries,
                           ec='k',
                           fc='w')
    ax.add_patch(patch_rect)

    # Add a colorbar over the rectangle:
    factor_cbar_width = 0.8
    x_cbar_start = x_rect_start + ((1 - factor_cbar_width) *
                                   width_rect /
                                   2)
    width_cbar = (width_rect * factor_cbar_width)
    y_cbar_start = (y_rect_start + 0.075)
    height_cbar = 0.04
    ax_cbar = ax.inset_axes(bounds=[x_cbar_start,
                                    y_cbar_start,
                                    width_cbar,
                                    height_cbar])
    cbar = fig_top.colorbar(im_b_scan,
                            cax=ax_cbar,
                            ax=ax,
                            ticks=c_bar_ticks,
                            orientation='horizontal')
    cbar.ax.tick_params(labelsize=fontsize_labels_pt, pad=1)
    if cbar_tick_labels:
        cbar.ax.set_xticks(c_bar_ticks, labels=cbar_tick_labels)
    # Label the colorbar with text:
    ax.text(0.5,
            (y_rect_start + height_rect - 0.015),
            string_c_bar_label,
            fontsize=fontsize_labels_pt,
            transform=ax.transAxes,
            ha='center',
            va='top'
            )


def plot_waveset_subfigure(
    fig,
    corr_array_exp,
    corr_array_theo,
    thetas_to_line,
    tuple_abc_labels,
    list_of_hilbert_sums_exp,
    list_of_hilbert_sums_theo,
    list_of_sum_labels,
    sum_label_has_exp,
    sum_label_has_theo,
    label_exp,
    label_theo,
    top_of_page=False
):
    # Split figure into four subplot axes:
    ((ax_corr_exp, ax_sums_exp),
     (ax_corr_theo, ax_sums_theo)
     ) = fig.subplots(2, 2)

    # Experimental NMO-corrected array plot:
    im_corr_exp = ax_corr_exp.imshow(corr_array_exp,
                                     cmap=c_map,
                                     extent=extent_corr,
                                     aspect='auto')
    im_corr_exp.set_clim(vmin=v_min_exp_nm,
                         vmax=v_max_exp_nm)
    corr_plot_ax_format(ax_corr_exp)
    wave_set_label(ax_corr_exp,
                   label_exp)

    # Experimental sum wiggle plot:
    plot_wiggles_DAS_central_20_percent(ax_sums_exp,
                                        list_of_hilbert_sums_exp,
                                        z_apex_vector_mm,
                                        list_of_sum_labels,
                                        x_axis_amplitude_wiggle_plots,
                                        cmap_extreme_pos,
                                        cmap_extreme_neg,
                                        True,
                                        True,
                                        sum_label_has_exp)
    ax_sums_exp.set_ylim(ax_corr_exp.get_ylim())

    if top_of_page:
        ax_corr_exp.set_xlabel(r'Ray angle $\theta$ (°)',
                               fontsize=fontsize_labels_pt)
        ax_sums_exp.set_xlabel('Sum waveforms',
                               fontsize=fontsize_labels_pt)

    if thetas_to_line:
        for theta_deg in thetas_to_line:
            if theta_deg is theta_crit_deg:
                label = r' $\theta$*'
            else:
                label = f' {theta_deg}°'
            theta_lines(ax_corr_exp, theta_deg, label=label)
            theta_lines(ax_corr_theo, theta_deg)

    # Label x_SDH:
    x_line(ax_corr_exp, params_al_sdh.x_sdh_mm, r' $x_\mathrm{SDH}$')
    x_line(ax_corr_theo, params_al_sdh.x_sdh_mm)

    # Theoretical corr array plot:
    im_corr_theo = ax_corr_theo.imshow(corr_array_theo,
                                       cmap=c_map,
                                       extent=extent_corr,
                                       aspect='auto')
    im_corr_theo.set_clim(vmin=v_min_theo_m,
                          vmax=v_max_theo_m)
    corr_plot_ax_format(ax_corr_theo)
    wave_set_label(ax_corr_theo, label_theo)
    ax_corr_theo.tick_params(labeltop=False)

    # Theoretical sums wiggle plot:
    plot_wiggles_DAS_central_20_percent(ax_sums_theo,
                                        list_of_hilbert_sums_theo,
                                        z_apex_vector_mm,
                                        list_of_sum_labels,
                                        x_axis_amplitude_wiggle_plots,
                                        cmap_extreme_pos,
                                        cmap_extreme_neg,
                                        True,
                                        True,
                                        sum_label_has_theo,
                                        central_percent=40)
    ax_sums_theo.set_ylim(ax_corr_theo.get_ylim())
    ax_sums_theo.tick_params(labeltop=False)

    # Set abc labels:
    axis_label(ax_corr_exp, f'({tuple_abc_labels[0]})')
    axis_label(ax_sums_exp, f'({tuple_abc_labels[1]})')
    axis_label(ax_corr_theo, f'({tuple_abc_labels[2]})')
    axis_label(ax_sums_theo, f'({tuple_abc_labels[3]})')


# %% Generate plots:

# Make top-level figure:
fig_top = plt.figure(figsize=(fig_width_inches, fig_height_inches),
                     layout="constrained")
# Split top-level figure into left and right subfigures:
fig_left, fig_right = fig_top.subfigures(1, 2, width_ratios=[1, 1.8])

# For the left figure: split vertically into three subplots:
(ax_ray_diagram,
 ax_b_scan_exp,
 ax_b_scan_theo
 ) = fig_left.subplots(3, 1, height_ratios=[0.32, 1, 1])


# Plot ray diagram in top left axes:
plot_ray_diagram_sdh_mss(ax_ray_diagram, '(a)')


# Plot an imshow of the experimental B-scan displacements
# in picometres:
plot_b_scan(ax_b_scan_exp,
            fig_top,
            mss_b_scan_nm * 1000,
            v_min_exp_nm * 1000,
            v_max_exp_nm * 1000,
            cbar_ticks_exp_pm,
            'Displacement $u_z$ (pm)',
            label_x=True
            )
axis_label(ax_b_scan_exp, '(b)')

# Plot an imshow of the theoretical B-scan displacements:
plot_b_scan(ax_b_scan_theo,
            fig_top,
            u_z_ascans_sdh_bernspice_m.T,
            v_min_theo_m,
            v_max_theo_m,
            cbar_ticks_theo,
            'Displacement $u_z$ (Arb.)',
            ['-', '0', '+']
            )
axis_label(ax_b_scan_theo, '(c)')


# In the right figure, split vertically into 3 subfigures:
(fig_ll,
 fig_ss,
 fig_hs) = fig_right.subfigures(3, 1, hspace=0.1)

# Wave set plots:

# L-L subfigure:
plot_waveset_subfigure(fig_ll,
                       corr_LL_exp_nm,
                       corr_LL_theo,
                       None,
                       ('d', 'e', 'f', 'g'),
                       [hilbert(sum_LL_all_exp_nm)],
                       [hilbert(sum_LL_all_theo_norm)],
                       ['0-64'],
                       ['left'],
                       ['left'],
                       (r'L-L $c_\mathrm{L}$=' +
                        f'{material_al_sdh.c_L_mpers}' +
                        r'$\,$m/s'),
                       'L-L modelled',
                       True)

# Add text in the theoretical L-L wiggle plot
# stating that the peak of the envelope function was
# artificially normalised to the experimental value:
string_ll_theo_message = (
    'Envelope peak\n' +
    'scaled to\n' +
    'experimental\n' +
    'value.'
)
fig_top.text(0.99, 0.75,
             string_ll_theo_message,
             fontsize=fontsize_labels_pt,
             ma='right',
             ha='right')

# S-S subfigure:
plot_waveset_subfigure(fig_ss,
                       corr_SS_exp_nm,
                       corr_SS_theo,
                       [theta_crit_deg, 50, 60],
                       ('h', 'i', 'j', 'k'),
                       [hilbert(sum_SS_0_crit_exp_nm),
                        hilbert(sum_SS_50_60_exp_nm),
                        hilbert(sum_SS_0_crit_plus_50_60_exp_nm),
                        hilbert(sum_SS_0_60_exp_nm)],
                       [hilbert(sum_SS_0_crit_theo_norm),
                        hilbert(sum_SS_50_60_theo_norm),
                        hilbert(sum_SS_0_crit_plus_50_60_theo_norm),
                        hilbert(sum_SS_0_60_theo_norm)],
                       [r'0-$\theta$*',
                        r'50-60',
                        r'0-$\theta$*+50-60',
                        r'0-60'],
                       ['left', 'left', 'left', -0.47],
                       [-0.25, -0.45, 'right', -1.2],
                       (r'S-S $c_\mathrm{S}$=' +
                        f'{material_al_sdh.c_T_mpers}' +
                        r'$\,$m/s'),
                       'S-S modelled')

# H-S subfigure:
plot_waveset_subfigure(fig_hs,
                       corr_hybrid_S_exp_nm,
                       corr_hybrid_S_theo,
                       [theta_crit_deg, 60],
                       ('l', 'm', 'n', 'o'),
                       [hilbert(sum_HS_crit_60_exp_nm),
                        hilbert(sum_hybrid_S_0_60_exp_nm)],
                       [hilbert(sum_HS_crit_60_theo_norm),
                        hilbert(sum_hybrid_S_0_60_theo_norm)],
                       [r'$\theta$*-60',
                        '0-60'],
                       ['right', 'right'],
                       ['right', -1.00],
                       (r'Hybrid-S $c_\mathrm{LSAW}$=' +
                        f'{c_LSW_mpers}' +
                        r'$\,$m/s'),
                       'Hybrid-S modelled'
                       )

# %% Save figure to file:

# Saving:
# fig_top.savefig(r'Figure folders\Fig Al SDH DAS\SDH_DAS_BernSpice_JASA.pdf',
#                 format='pdf', dpi=600)

# fig_top.savefig(r'Figure folders\Fig Al SDH DAS\SDH_DAS_BernSpice_JASA.png',
#                 format='png', dpi=200)

# %%
