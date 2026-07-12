# %%
import matplotlib.pyplot as plt
import colorcet as cc
import numpy as np
from scipy.signal import hilbert

from corevariables.modhfsexperiment import params_al_nmo
from functions.modnmocorrection import nmo_correct_b_vector
from functions.modhwcorrect import cdp_head_wave_correct
from functions.modthetacrittrig import calculate_theta_crit_deg
from functions.modsumgather import sum_gather
from functions.modmyplotfuncs import (plot_wiggles_DAS_central_20_percent)
from corevariables.modjasa import (fontsize_labels_pt,
                                   width_fig_two_column_jasa_in)
from functions.modmyplotfuncs import set_mpl_fonts_Times_New_Roman
from functions.modraydiagramnmo import plot_ray_diagram_nmo
from bernspice_hfs import get_bernspice_a_scans_hfs
from functions.modcalculationsfortable import get_table_row_hfs

# %%

# IPython magic:
# %load_ext autoreload
# %autoreload 2

# Load the experimental data:

# Convert the displacement data from pm to nm:
b_scan_ds_nm = params_al_nmo.b_scan_array_2d_detrend_pm / 1000

# Generate axis vectors:
x_max_mm = params_al_nmo.x_max_mm
x_vector_mm = params_al_nmo.x_vector_mm
t_min_us = params_al_nmo.t_min_us
t_max_us = params_al_nmo.t_max_us
rise_time_us = params_al_nmo.rise_time_us
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
angle_spec_ray_max_deg = np.rad2deg(np.atan2(x_max_mm * 10**-3,
                                             2 * b_mm * 10**-3))
c_L_mpers = 6371  # Nemo
c_T_mpers = 3077  # Nemo
theta_crit_deg = calculate_theta_crit_deg(c_L_mpers, c_T_mpers)
x_crit_mm = np.tan(np.deg2rad(theta_crit_deg)) * 2 * b_mm
# Calculate the x values associated with 15 -degree intervals up to 90 degrees:
theta_max_deg = np.rad2deg(np.atan2(x_max_mm, 2 * b_mm))
tick_angles_spaced_deg = np.linspace(0, 60, 5)
tick_angles_deg = np.append(tick_angles_spaced_deg, theta_max_deg)
x_ticks_mm = np.tan(np.deg2rad(tick_angles_deg)) * 2 * b_mm
labels_angles = [f'{angle:.0f}' for angle in tick_angles_deg]

# Get NMO corrected extracts of the experimental data:
t_0_TT_us = 1.28
t_0_half_width_us = 0.4

# Define a vector of sample thickness values (b) to use
# as a common 'imaging grid' for all of the different
# wave sets:
b_min_mm = (c_T_mpers *
            (((t_0_TT_us - t_0_half_width_us) / 2) *
             10**-6)
            ) / (10**-3)
b_max_mm = (c_T_mpers *
            (((t_0_TT_us + t_0_half_width_us) / 2) *
             10**-6)
            ) / (10**-3)
n_bs = 200
b_vector_mm = np.linspace(b_min_mm, b_max_mm, n_bs)

# L-L wavefront:
corr_LL_exp_nm = nmo_correct_b_vector(b_scan_ds_nm,
                                      time_vector_us,
                                      b_vector_mm,
                                      x_vector_mm,
                                      c_L_mpers)

# T-T wavefront:
corr_SS_exp_nm = nmo_correct_b_vector(b_scan_ds_nm,
                                      time_vector_us,
                                      b_vector_mm,
                                      x_vector_mm,
                                      c_T_mpers)

# H-T wavefront:
c_LSW_mpers = 6000
b_scan_HS_corr_exp_nm = cdp_head_wave_correct(b_scan_ds_nm,
                                              b_vector_mm,
                                              x_vector_mm,
                                              time_vector_us,
                                              c_T_mpers,
                                              c_LSW_mpers)

# Create the hybrid-corrected wavefront by using the S-S direct correction up
# until theta_crit, then using the H-T correction from theta-crit upwards.
# Calculate the indices in the corrected arrays associated with the critical
# angle:
index_first_supercrit = np.argmax(angles_spec_rays_vector_deg > theta_crit_deg)
# Slice corrected arrays:
hybrid_corr_SS_part_nm = corr_SS_exp_nm[:, 0:index_first_supercrit]
hybrid_corr_HS_part_nm = b_scan_HS_corr_exp_nm[:, index_first_supercrit:]
# Concatenate:
corr_hybrid_S_exp_nm = np.hstack(
    (hybrid_corr_SS_part_nm, hybrid_corr_HS_part_nm))

# Sum delayed gathers over certain angular ranges:
sum_LL_all_exp_nm = sum_gather(
    corr_LL_exp_nm, angles_spec_rays_vector_deg)

sum_ds_TT_all_nm = sum_gather(
    corr_SS_exp_nm, angles_spec_rays_vector_deg)  # SAW crosstalk
sum_SS_0_crit_exp_nm = sum_gather(corr_SS_exp_nm,
                                  angles_spec_rays_vector_deg,
                                  theta_max_deg=theta_crit_deg)
sum_SS_crit_48_exp_nm = sum_gather(corr_SS_exp_nm,
                                   angles_spec_rays_vector_deg,
                                   theta_start_deg=theta_crit_deg,
                                   theta_max_deg=48)
sum_SS_48_64_exp_nm = sum_gather(corr_SS_exp_nm,
                                 angles_spec_rays_vector_deg,
                                 theta_start_deg=48,
                                 theta_max_deg=64)
sum_SS_0_64_exp_nm = sum_gather(corr_SS_exp_nm,
                                angles_spec_rays_vector_deg,
                                theta_max_deg=64)

sum_ds_HS_supercrit_nm = sum_gather(b_scan_HS_corr_exp_nm,
                                    angles_spec_rays_vector_deg,
                                    theta_start_deg=theta_crit_deg)
sum_ds_HS_all_nm = sum_gather(
    b_scan_HS_corr_exp_nm, angles_spec_rays_vector_deg)

sum_ds_hybrid_all_nm = sum_gather(
    corr_hybrid_S_exp_nm, angles_spec_rays_vector_deg)

# %% Run the theoretical model:
# Bernstein & Spicer model with reflection & detection coefficients:

# Generate synthetic A-scans:
n_times_theoretical = 4000
time_vector_theoretical_us = np.linspace(t_min_us,
                                         t_max_us,
                                         n_times_theoretical)
# A-scans returned in rows of ndarray.
(u_z_ascans_hfs_bernspice_m
 ) = get_bernspice_a_scans_hfs(time_vector_theoretical_us * 10**-6)

# Get NMO-corrected extracts:
# L-L wavefront:
corr_LL_theory_m = nmo_correct_b_vector(u_z_ascans_hfs_bernspice_m.T,
                                        time_vector_theoretical_us,
                                        b_vector_mm,
                                        x_vector_mm,
                                        c_L_mpers)

# T-T wavefront:
corr_SS_theory_m = nmo_correct_b_vector(u_z_ascans_hfs_bernspice_m.T,
                                        time_vector_theoretical_us,
                                        b_vector_mm,
                                        x_vector_mm,
                                        c_T_mpers)

# H-T wavefront:
b_scan_HS_corr_theory_m = cdp_head_wave_correct(u_z_ascans_hfs_bernspice_m.T,
                                                b_vector_mm,
                                                x_vector_mm,
                                                time_vector_theoretical_us,
                                                c_T_mpers,
                                                c_L_mpers)

# Theoretical: Create the hybrid-corrected wavefront by using the S-S direct
# correction up until theta_crit, then using the H-T correction from
# theta-crit upwards.
# Calculate the indices in the corrected arrays associated with the critical
# angle:
# Slice corrected arrays:
hybrid_corr_theory_SS_part_m = corr_SS_theory_m[:,
                                                0:index_first_supercrit]
hybrid_corr_theory_HS_part_m = b_scan_HS_corr_theory_m[:,
                                                       index_first_supercrit:]
# Concatenate:
corr_hybrid_S_theo_m = np.hstack((hybrid_corr_theory_SS_part_m,
                                  hybrid_corr_theory_HS_part_m))

# Sum the corrected gathers over certain angular ranges:
sum_ds_LL_all_theory_m = sum_gather(corr_LL_theory_m,
                                    angles_spec_rays_vector_deg)

sum_ds_TT_all_theory_m = sum_gather(corr_SS_theory_m,
                                    angles_spec_rays_vector_deg)
sum_ds_TT_0_crit_theory_m = sum_gather(corr_SS_theory_m,
                                       angles_spec_rays_vector_deg,
                                       theta_max_deg=theta_crit_deg)
sum_ds_TT_crit_48_theory_m = sum_gather(corr_SS_theory_m,
                                        angles_spec_rays_vector_deg,
                                        theta_start_deg=theta_crit_deg,
                                        theta_max_deg=48)
sum_ds_TT_48_64_theory_m = sum_gather(corr_SS_theory_m,
                                      angles_spec_rays_vector_deg,
                                      theta_start_deg=48,
                                      theta_max_deg=64)
sum_ds_TT_0_64_theory_m = sum_gather(corr_SS_theory_m,
                                     angles_spec_rays_vector_deg,
                                     theta_max_deg=64)

sum_ds_HS_supercrit_theory_m = sum_gather(b_scan_HS_corr_theory_m,
                                          angles_spec_rays_vector_deg,
                                          theta_start_deg=theta_crit_deg)

sum_ds_hybrid_all_theory_m = sum_gather(corr_hybrid_S_theo_m,
                                        angles_spec_rays_vector_deg)

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
                     np.max(np.abs(hilbert(sum_ds_LL_all_theory_m)))
                     )

sum_LL_all_theo_norm = (sum_ds_LL_all_theory_m * scale_factor_theo)

# Apply the same scale factor to all the other theoretical sum
# waveforms:
# S-S:
sum_SS_0_crit_theo_norm = sum_ds_TT_0_crit_theory_m * scale_factor_theo
sum_SS_crit_48_theo_norm = sum_ds_TT_crit_48_theory_m * scale_factor_theo
sum_SS_48_64_theo_norm = sum_ds_TT_48_64_theory_m * scale_factor_theo
sum_SS_0_64_theo_norm = sum_ds_TT_0_64_theory_m * scale_factor_theo

# Head-S:
sum_HS_supercrit_theo_norm = sum_ds_HS_supercrit_theory_m * scale_factor_theo

# Hybrid-S:
sum_hybrid_S_all_theo_norm = sum_ds_hybrid_all_theory_m * scale_factor_theo


# %% Get rows of the results table:

theta_sep_deg = 48

# Experimental:
row_ll_exp = get_table_row_hfs(corr_LL_exp_nm,
                               angles_spec_rays_vector_deg,
                               theta_crit_deg,
                               theta_sep_deg,
                               None,
                               20)
row_ss_exp = get_table_row_hfs(corr_SS_exp_nm,
                               angles_spec_rays_vector_deg,
                               theta_crit_deg,
                               theta_sep_deg,
                               64,
                               20)
row_hybrid_s_exp = get_table_row_hfs(corr_hybrid_S_exp_nm,
                                     angles_spec_rays_vector_deg,
                                     theta_crit_deg,
                                     theta_sep_deg,
                                     None,
                                     40)

# Modelled:
row_ll_theo = get_table_row_hfs(corr_LL_theory_m * scale_factor_theo,
                                angles_spec_rays_vector_deg,
                                theta_crit_deg,
                                theta_sep_deg,
                                None,
                                40)
row_ss_theo = get_table_row_hfs(corr_SS_theory_m * scale_factor_theo,
                                angles_spec_rays_vector_deg,
                                theta_crit_deg,
                                theta_sep_deg,
                                64,
                                40)
row_hybrid_s_theo = get_table_row_hfs(corr_hybrid_S_theo_m * scale_factor_theo,
                                      angles_spec_rays_vector_deg,
                                      theta_crit_deg,
                                      theta_sep_deg,
                                      None,
                                      40)

# Stack rows into arrays that you can view in the debugger:
table_array_exp = np.vstack((row_ll_exp,
                             row_ss_exp,
                             row_hybrid_s_exp))
table_array_theo = np.vstack((row_ll_theo,
                              row_ss_theo,
                              row_hybrid_s_theo))

# %% Define plotting parameters & functions:

# Plotting parameters:
fig_width_inches = width_fig_two_column_jasa_in
fig_height_inches = 4/3 * 6 * 0.9
fig_dpi = 200
set_mpl_fonts_Times_New_Roman()

# Set displacement colormap limits:
# Experimental displacement colormap limits:
v_min_exp_nm = -0.1
v_max_exp_nm = 0.1
cbar_ticks_exp_pm = [-100, 0, 100]
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
x_axis_amplitude_wiggles_exp = 32
# If the normalisation process has worked correctly,
# we should be able to use the same x scale for exp and theo
# wiggle plots:
x_axis_amplitude_wiggles_theoretical = x_axis_amplitude_wiggles_exp
# Extent for b-scan imshows:
extent_b_scan = [0, x_max_mm,
                 t_max_us-rise_time_us, t_min_us-rise_time_us]
# Extent for corr array imshows:
extent_corr = [0, x_max_mm,
               b_max_mm, b_min_mm]

# Define plotting functions:


def common_axes_format(ax):
    # ax.set_ylabel(r'Time $t$ (μs)', fontsize=fontsize_labels_pt)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax.xaxis.set_label_position('top')


def corr_plot_axes_format(ax):
    common_axes_format(ax)
    # Angle varies as the arctangent of the x_offset over b.
    # We can use non-periodic tick spacing to mark 10 degree angle intervals.
    ax.set_xticks(x_ticks_mm, labels=labels_angles)
    ax.set_yticks([1.5, 2.0, 2.5])
    # Label y-axis:
    ax.set_ylabel(r'$b$ (mm)', fontsize=fontsize_labels_pt)


def theta_line(ax, angle_deg, label):
    # Plot a dashed axvline at the x value corresponding to the given angle:
    x_theta_mm = np.tan(np.deg2rad(angle_deg)) * 2 * b_mm
    ax.axvline(x_theta_mm, color=(1, 1, 1), dashes=(4, 4), linewidth=0.7)
    t_lims = ax.get_ylim()
    t_text = t_lims[1] + ((t_lims[0] - t_lims[1]) * 0.96)
    ax.text(x_theta_mm, t_text, label, color=(1, 1, 1),
            fontsize=fontsize_labels_pt)


def axis_label(ax, label, pad=-9):
    ax.set_title(label, fontsize=fontsize_labels_pt, y=0, pad=pad)


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
    common_axes_format(ax)
    ax.set_ylim(3.5, -0.25)
    if label_x:
        ax.set_xlabel(r'Off-set $x$ (mm)', fontsize=fontsize_labels_pt)
    else:
        ax.set_xticks(ax.get_xticks(), [])
    ax.set_ylabel(r'Time $t$ (μs)', fontsize=fontsize_labels_pt)

    # Add a colorbar inset into the top-right corner of the B-scan plot:
    ax_cbar = ax.inset_axes(bounds=[0.89, 0.7, 0.06, 0.25])
    cbar = fig_top.colorbar(im_b_scan,
                            cax=ax_cbar,
                            ax=ax,
                            ticks=c_bar_ticks,
                            orientation='vertical')
    cbar.ax.tick_params(left=True, labelleft=True,
                        right=False, labelright=False,
                        labelsize=fontsize_labels_pt)
    if cbar_tick_labels:
        cbar.ax.set_yticks(c_bar_ticks, labels=cbar_tick_labels)
    # Label the colorbar with text:
    ax.text(0.72,
            0.96,
            string_c_bar_label,
            fontsize=fontsize_labels_pt,
            transform=ax.transAxes,
            ha='right',
            va='top',
            ma='right'
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
    corr_plot_axes_format(ax_corr_exp)
    wave_set_label(ax_corr_exp,
                   label_exp)

    # Experimental sum wiggle plot:
    plot_wiggles_DAS_central_20_percent(ax_sums_exp,
                                        list_of_hilbert_sums_exp,
                                        b_vector_mm,
                                        list_of_sum_labels,
                                        x_axis_amplitude_wiggles_exp,
                                        cmap_extreme_pos,
                                        cmap_extreme_neg,
                                        True,
                                        True,
                                        sum_label_has_exp)
    ax_sums_exp.set_ylim(ax_corr_exp.get_ylim())

    if top_of_page:
        ax_corr_exp.set_xlabel(r'Specular angle $\theta$ (°)',
                               fontsize=fontsize_labels_pt)
        ax_sums_exp.set_xlabel('Sum waveforms',
                               fontsize=fontsize_labels_pt)

    if thetas_to_line:
        for theta_deg in thetas_to_line:
            if theta_deg is theta_crit_deg:
                label = r' $\theta$*'
            else:
                label = f' {theta_deg}°'
            theta_line(ax_corr_exp, theta_deg, label)
            theta_line(ax_corr_theo, theta_deg, '')

    # Theoretical corr array plot:
    im_corr_theo = ax_corr_theo.imshow(corr_array_theo,
                                       cmap=c_map,
                                       extent=extent_corr,
                                       aspect='auto')
    im_corr_theo.set_clim(vmin=v_min_theo_m,
                          vmax=v_max_theo_m)
    corr_plot_axes_format(ax_corr_theo)
    wave_set_label(ax_corr_theo, label_theo)
    ax_corr_theo.tick_params(labeltop=False)

    # Theoretical sums wiggle plot:
    plot_wiggles_DAS_central_20_percent(ax_sums_theo,
                                        list_of_hilbert_sums_theo,
                                        b_vector_mm,
                                        list_of_sum_labels,
                                        x_axis_amplitude_wiggles_theoretical,
                                        cmap_extreme_pos,
                                        cmap_extreme_neg,
                                        True,
                                        True,
                                        sum_label_has_theo)
    ax_sums_theo.set_ylim(ax_corr_theo.get_ylim())
    ax_sums_theo.tick_params(labeltop=False)

    # Set abc labels:
    axis_label(ax_corr_exp, f'({tuple_abc_labels[0]})')
    axis_label(ax_sums_exp, f'({tuple_abc_labels[1]})')
    axis_label(ax_corr_theo, f'({tuple_abc_labels[2]})')
    axis_label(ax_sums_theo, f'({tuple_abc_labels[3]})')


# %% Plotting:

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
plot_ray_diagram_nmo(ax_ray_diagram, x_max_mm, b_mm)
axis_label(ax_ray_diagram, '(a)', pad=-2)


# Plot an imshow of the experimental B-scan displacements
# in picometres:
plot_b_scan(ax_b_scan_exp,
            fig_top,
            b_scan_ds_nm*1000,
            v_min_exp_nm*1000,
            v_max_exp_nm*1000,
            cbar_ticks_exp_pm,
            'Displacement\n$u_z$ (pm)',
            label_x=True
            )
axis_label(ax_b_scan_exp, '(b)')


# Plot theoretical B-scan on the left:
plot_b_scan(ax_b_scan_theo,
            fig_top,
            u_z_ascans_hfs_bernspice_m.T,
            v_min_theo_m,
            v_max_theo_m,
            cbar_ticks_theo,
            'Displacement\n$u_z$ (Arb.)',
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
                       corr_LL_theory_m,
                       None,
                       ('d', 'e', 'f', 'g'),
                       [hilbert(sum_LL_all_exp_nm)],
                       [hilbert(sum_LL_all_theo_norm)],
                       ['0-68'],
                       ['left'],
                       ['left'],
                       r'L-L $c_\mathrm{L}$=' + f'{c_L_mpers}' + r'$\,$m/s',
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
                       corr_SS_theory_m,
                       [theta_crit_deg, 48, 64],
                       ('h', 'i', 'j', 'k'),
                       [hilbert(sum_SS_0_crit_exp_nm),
                        hilbert(sum_SS_crit_48_exp_nm),
                        hilbert(sum_SS_48_64_exp_nm),
                        hilbert(sum_SS_0_64_exp_nm)],
                       [hilbert(sum_SS_0_crit_theo_norm),
                        hilbert(sum_SS_crit_48_theo_norm),
                        hilbert(sum_SS_48_64_theo_norm),
                        hilbert(sum_SS_0_64_theo_norm)],
                       [r'0-$\theta$*',
                        r'$\theta$*-48',
                        '48-64',
                        '0-64'],
                       ['left', 'left', 0.0, -2.0],
                       ['left', 'right', 'left', 'right'],
                       r'S-S $c_\mathrm{S}$=' + f'{c_T_mpers}' + r'$\,$m/s',
                       'S-S modelled')

# H-S subfigure:
plot_waveset_subfigure(fig_hs,
                       corr_hybrid_S_exp_nm,
                       corr_hybrid_S_theo_m,
                       [theta_crit_deg],
                       ('l', 'm', 'n', 'o'),
                       [hilbert(sum_ds_HS_supercrit_nm),
                        hilbert(sum_ds_hybrid_all_nm)],
                       [hilbert(sum_HS_supercrit_theo_norm),
                        hilbert(sum_hybrid_S_all_theo_norm)],
                       [r'$\theta$*-68', r'0-68'],
                       ['left', 0.0],
                       ['left', 0.0],
                       (r'Hybrid-S $c_\mathrm{LSAW}$=' +
                        f'{c_LSW_mpers}' +
                        r'$\,$m/s'),
                       'Hybrid-S modelled'
                       )


# %% Saving figure to file:

# Saving:
# fig_top.savefig(
#     r'Figure folders\Fig Al NMO DAS\NMO_DAS_BernSpice_JASA.pdf',
#     format='pdf',
#     dpi=600)

# fig_top.savefig(
#     r'Figure folders\Fig Al NMO DAS\NMO_DAS_BernSpice_JASA.png',
#     format='png',
#     dpi=200)

# %%
