# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from corevariables.modaluminium import material_Al
from corevariables.modjasa import (width_fig_one_column_jasa_in,
                                   fontsize_labels_pt,
                                   my_dashes)
from functions.modmyplotfuncs import set_mpl_fonts_Times_New_Roman
from functions.modbuildxgenandxdetmatrices import build_x_gen_and_x_det_matrices_m
from functions.delaylawfunctions.moddelaylawfunctions import calculate_delay_law_tt
from functions.modcreatecheopsmesh import create_cheops_meshes


# Define figure parameters:
fig_width_in = width_fig_one_column_jasa_in
fig_height_in = 2.4
fig_dpi = 200
set_mpl_fonts_Times_New_Roman()

# Ray diagram geometry:
x_max_mm = 10
z_max_mm = 5
x_array_start_mm = 1
x_array_end_mm = x_max_mm - x_array_start_mm
n_elements = 32
pitch_mm = (x_array_end_mm - x_array_start_mm) / (n_elements - 1)
x_elements_mm = np.linspace(x_array_start_mm, x_array_end_mm, n_elements)
g_index = 5
d_index = 24
x_gen_mm = x_elements_mm[g_index]
x_det_mm = x_elements_mm[d_index]
laser_line_length_mm = 1
lw_lasers = 1
lw_wavefronts = 1
lw_boundaries = 1

xz_gen_mm = (x_gen_mm, 0)
x_scatterer_mm = 5.5
z_scatterer_mm = 3
xz_scatterer_mm = (x_scatterer_mm, z_scatterer_mm)
r_sdh_mm = 0.1
theta_rad = np.atan2((x_scatterer_mm - x_gen_mm),
                     z_scatterer_mm)

# Build locations of send and receive ray heads:
factor_send = 0.8
x_send_head_mm = x_gen_mm + ((x_scatterer_mm - x_gen_mm) * factor_send)
z_send_head_mm = z_scatterer_mm * factor_send
xz_send_head_mm = (x_send_head_mm, z_send_head_mm)

# Find x and z coordinates of return ray head:
factor_receive = 0.7
x_return_head_mm = (x_scatterer_mm +
                    (factor_receive * (x_det_mm - x_scatterer_mm)))
z_return_head_mm = (z_scatterer_mm -
                    (factor_receive * z_scatterer_mm))
xz_return_head_mm = (x_return_head_mm, z_return_head_mm)

# Calculate pre- and post-reflection wavefront radii:
r_wavefront_pre_reflection_mm = np.sqrt((x_send_head_mm - x_gen_mm)**2 +
                                        z_send_head_mm**2)
r_wavefront_post_reflection_mm = np.sqrt((x_return_head_mm -
                                          x_scatterer_mm)**2 +
                                         (z_scatterer_mm -
                                          z_return_head_mm)**2)

# Compute the Cheops pyramid for the scatterer location, array and material:
(x_gen_matrix_m,
 x_det_matrix_m) = build_x_gen_and_x_det_matrices_m(n_elements, pitch_mm)
# Get scatterer coordinates relative to centre of array:
dx_scatterer_m = (x_scatterer_mm - 5) * 10**-3
delay_matrix_s = calculate_delay_law_tt(dx_scatterer_m,
                                        z_scatterer_mm*10**-3,
                                        x_gen_matrix_m, x_det_matrix_m,
                                        v_t_mpers=material_Al.c_t_mpers)

# Create gen and det index meshes for Cheops 3d plotting:
gen_indices_mesh, det_indices_mesh = create_cheops_meshes(n_elements)

# Extract monostatic delays (g=d) from Cheops Pyramid:
delay_vector_monostatic_s = np.diagonal(delay_matrix_s)

# Compute delays for a common mid-point (CMP) scan where the common mid point
# is x_scatterer:
max_offset_cmp_m = pitch_mm * (n_elements - 1) * 10**-3
offsets_cmp_m = np.linspace(0, max_offset_cmp_m, n_elements)
delay_vector_cmp_s = ((1/material_Al.c_t_mpers) *
                      np.sqrt((z_scatterer_mm * 10**-3)**2 + offsets_cmp_m**2))

# Aesthetic parameters:
dict_arrowprops_arrow = dict(arrowstyle='->', shrinkA=0, shrinkB=0,
                             lw=1)
# Set 3D grid line colours to black before plotting surface:
plt.rcParams['grid.color'] = "k"
# Title fontsize:
fontsize_titles = fontsize_labels_pt

# Define plotting functions:


def set_colors(everything_black=False):
    if everything_black:
        c_post_reflection_wavefront = 'k'
        c_pre_reflection_wavefront = 'k'
        c_array = 'k'
        c_gen_laser = 'k'
        c_det_laser = 'k'
    else:
        c_post_reflection_wavefront = 'limegreen'
        c_pre_reflection_wavefront = 'cyan'
        c_array = 'magenta'
        c_gen_laser = 'red'
        c_det_laser = 'lime'

    return (c_post_reflection_wavefront,
            c_pre_reflection_wavefront,
            c_array, c_gen_laser, c_det_laser)


(c_post_reflection_wavefront,
 c_pre_reflection_wavefront,
 c_array, c_gen_laser, c_det_laser) = set_colors(True)


def plot_ray_diagram_circulars(ax, axis_label_string):
    # Format original axis:
    ax.set_aspect('equal')
    # Set axis limits:
    ax.set_xlim(0, x_max_mm)
    ax.set_ylim(z_max_mm, -laser_line_length_mm)
    # Hide the axes spines and ticks:
    ax.set_axis_off()
    # Label y-axis:
    ax.set_ylabel('Depth $z$ (mm)', fontsize=fontsize_labels_pt)

    # Draw rect representing solid sample, (0, 0) in top left corner:
    rect = patches.Rectangle((0, 0), x_max_mm, z_max_mm, linewidth=lw_boundaries,
                             clip_on=False, edgecolor='k', facecolor='none',
                             zorder=2)
    # Add gen laser:
    ax.vlines(x_gen_mm, 0, -laser_line_length_mm, c_gen_laser,
              linewidth=lw_lasers)
    # Add detection laser:
    ax.vlines(x_det_mm, 0, -laser_line_length_mm, c_det_laser,
              linewidth=lw_lasers)
    # Add circular pre-reflection wavefront:
    circ = patches.Arc(xz_gen_mm, 2 * r_wavefront_pre_reflection_mm,
                       2 * r_wavefront_pre_reflection_mm,
                       theta1=0, theta2=180,
                       edgecolor=c_pre_reflection_wavefront, facecolor='none',
                       lw=lw_wavefronts)
    ax.add_patch(circ)
    # Add circular post-reflection wavefront:
    circ = patches.Circle(xz_scatterer_mm, r_wavefront_post_reflection_mm,
                          edgecolor=c_post_reflection_wavefront, facecolor='none',
                          lw=lw_wavefronts)
    ax.add_patch(circ)
    # Add send arrow:
    ax.annotate('', xy=xz_send_head_mm,
                xytext=xz_gen_mm, xycoords='data',
                arrowprops=dict_arrowprops_arrow)
    # Add return arrow:
    ax.annotate('', xy=xz_return_head_mm,
                xytext=xz_scatterer_mm, xycoords='data',
                arrowprops=dict_arrowprops_arrow)
    # Add circle representing SDH:
    circ = patches.Circle(xz_scatterer_mm, r_sdh_mm, edgecolor='k', facecolor='k',
                          lw=lw_boundaries)
    ax.add_patch(circ)

    # Add array elements:
    ax.plot(x_elements_mm, np.zeros(n_elements) - 0.08, marker='s', mec='none',
            mfc=c_array, ls='none', ms=1.5)

    # Add rectangle last so that it is in front:
    ax.add_patch(rect)

    # Label g and d:
    ax.text(x_gen_mm, -laser_line_length_mm/2, '$g$  ', ha='right',
            fontsize=fontsize_labels_pt)
    ax.text(x_det_mm, -laser_line_length_mm/2.5, '$d$  ', ha='right',
            fontsize=fontsize_labels_pt)

    # Label x and z axes with little arrows:
    xz_xz_arrows_start = (0.5, 0.5)
    ax.annotate('', xytext=xz_xz_arrows_start, xy=(1.4, 0.5), xycoords='data',
                arrowprops=dict_arrowprops_arrow)
    ax.annotate('', xytext=xz_xz_arrows_start, xy=(0.5, 1.4), xycoords='data',
                arrowprops=dict_arrowprops_arrow)
    ax.text(1.3, 0.5, '$x$', fontsize=fontsize_labels_pt, va='center')
    ax.text(0.5, 1.25, '$z$', fontsize=fontsize_labels_pt, ha='center',
            va='top')

    # Label scatterer coordinates:
    ax.text(x_scatterer_mm, z_scatterer_mm + 0.3,
            r'($x_\mathrm{p}$, $z_\mathrm{p}$)', fontsize=fontsize_labels_pt,
            ha='center', va='top')

    # Add a letter label for the figure:
    ax.text(0.5, -0.15, axis_label_string, fontsize=fontsize_labels_pt,
            transform=ax.transAxes, ha='center')

    # Add title:
    ax.set_title('XZ plane:', fontsize=fontsize_titles, ha='center')


def plot_cheops_pyramid(ax_3d):
    # Plot the delay surface in 3D:
    ax_3d.plot_surface(gen_indices_mesh, det_indices_mesh, delay_matrix_s,
                       edgecolor='k', facecolor='w', alpha=1)
    # Format axes_3d:
    ax_3d.set_box_aspect((1, 1, 1))

    ax_3d.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax_3d.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax_3d.set_xticks([0, n_elements-1])
    ax_3d.set_yticks([0, n_elements-1])
    ax_3d.set_zticks([])
    # Invert z-axis:
    ax_3d.invert_zaxis()
    ax_3d.set_zlim([np.max(delay_matrix_s), np.min(delay_matrix_s)])
    # Invert y-axis for 'image-like' display:
    ax_3d.view_init(elev=18, azim=-110)
    ax_3d.set_xlim([0, n_elements - 1])
    ax_3d.set_ylim([0, n_elements - 1])
    ax_3d.invert_yaxis()
    # Set background pane colors:
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor('none')
    ax_3d.yaxis.pane.set_edgecolor('none')

    # # Custom sizing via set_position:
    # fig.canvas.draw()
    # # Position and scale the axes with ax.set_position([x, y, x_scale, y_scale]):
    # ax_3d.set_position([-0.25, -0.2, 1, 1])
    # # Set axis back-pane to be transparent in case it overlaps other subplots:
    # ax_3d.set_facecolor('none')

    # Sort out the axis lables:


def plot_delays_2d(ax, delay_vector_s):
    # Plot line:
    ax.plot(delay_vector_s, color='k', lw=1)
    # Set axis parameters:
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    n_delays = len(delay_vector_s)
    ax.set_xticks([0, n_delays-1])
    ax.set_yticks([])
    # Invert y-axis:
    ax.invert_yaxis()
    # Set axis limits:
    ax.set_xlim([0, n_delays-1])
    ax.set_ylim([np.max(delay_vector_s), np.min(delay_vector_s) - (1*10**-6)])
    # Set xlabel location:
    ax.xaxis.set_label_position('top')
    ax.xaxis.labelpad = -5
    # Set ylabel:
    ax.set_ylabel('Travel time\n', fontsize=fontsize_labels_pt)
    # Add an arrow to mark the direction of the time axis:
    ax.text(-0.15, 0.5, r'$\longleftarrow$', fontsize=fontsize_labels_pt + 6,
            rotation=90, ha='center', va='center', transform=ax.transAxes)
    # Hide the bottom spine:
    ax.spines[['bottom']].set_visible(False)
    # Set axes aspect ratio:
    ax.set_aspect(15*10**6)


def plot_delays_monostatic(ax):
    # Plot line:
    plot_delays_2d(ax, delay_vector_monostatic_s)
    # Set title:
    ax.set_title('Monostatic:', fontsize=fontsize_titles)
    ax.set_xlabel(r'$g=d$', fontsize=fontsize_labels_pt)
    ax.text(0.5, -0.15, '(b)', fontsize=fontsize_labels_pt,
            transform=ax.transAxes, ha='center')


def plot_delays_cmp(ax):
    # Plot line:
    plot_delays_2d(ax, delay_vector_cmp_s)
    # Set title:
    ax.set_title('CMP$=x_p$:', fontsize=fontsize_titles)
    ax.set_xlabel('Offset', fontsize=fontsize_labels_pt)
    ax.text(0.5, -0.15, '(c)', fontsize=fontsize_labels_pt,
            transform=ax.transAxes, ha='center')


# Create figure:
fig = plt.figure(figsize=(fig_width_in, fig_height_in), layout="constrained",
                 dpi=fig_dpi)
# Create sub-figures for arranging axes:
(fig_top, fig_bottom) = fig.subfigures(2, 1, height_ratios=[1, 1])

# Plot circular wavefront diagram in XZ plane:
ax_top = fig_top.subplots(1, 1)
plot_ray_diagram_circulars(ax_top, '(a)')

# # Plot Cheops pyramid in data volume on centre figure:
# ax_right = fig_center.add_subplot(projection='3d')
# plot_cheops_pyramid(ax_right)
# # Label 3D axes:
# fig_center.text(0.66, 0.14, 'Generation index $g$',
#                 fontsize=fontsize_labels_pt, ha='center')
# fig_center.text(0, 0.16, 'Detection\nindex $d$',
#                 fontsize=fontsize_labels_pt)
# fig_center.text(0.02, 0.52, 'Travel time',
#                 fontsize=fontsize_labels_pt, rotation=90, ha='center')
# # Add an arrow to mark the direction of the time axis:
# fig_center.text(0.08, 0.55, r'$\longleftarrow$',
#                 fontsize=fontsize_labels_pt + 6, rotation=90, ha='center')

# Plot slices:
ax_monostatic, ax_cmp = fig_bottom.subplots(1, 2)
plot_delays_monostatic(ax_monostatic)
plot_delays_cmp(ax_cmp)

# Add titles above sub-figures:
# height_titles = 1
# fig_top.text(0.5, height_titles, 'XZ plane ($x$, $z$):',
#              fontsize=fontsize_titles, ha='center')
# fig_center.text(0.5, height_titles, 'Full matrix volume ($g$, $d$, $t$):',
#                 fontsize=fontsize_titles, ha='center')

# fig.draw_without_rendering()

# # Write abc label beneath 3D axes:
# fig_center.text(0.5, 0.03, '(b)', fontsize=fontsize_labels_pt)

# %%
# Saving figure to image file:
# fig.savefig(r'Figure folders\Fig circular wavefront model\fig_CWM_JASA.pdf',
#             format='pdf', bbox_inches='tight')

# fig.savefig(r'Figure folders\Fig circular wavefront model\fig_CWM_JASA.png',
#             format='png', dpi=200, bbox_inches='tight')

# %%
