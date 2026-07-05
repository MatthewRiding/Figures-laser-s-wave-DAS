# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import colorcet as cc
import matplotlib.patches as patches
from matplotlib.colors import to_rgba

from functions.modthetacrittrig import (calculate_theta_crit_deg,
                                        tan_theta_crit)
from corevariables.modjasa import (width_fig_one_column_jasa_in,
                                   fontsize_labels_pt,
                                   my_dashes,
                                   line_width_plots,
                                   dict_arrowprops_arrow)
from functions.modmyplotfuncs import set_mpl_fonts_Times_New_Roman
import bernstein_and_spicer_model as bernspice


# IPython magic:
# %matplotlib tk
%load_ext autoreload
%autoreload 2

# Input parameters:
global_length_scale_factor = 1
# Grid parameters:
x_min_mm = 0
x_max_mm = 15 * global_length_scale_factor
z_max_mm = 10 * global_length_scale_factor
n_pixels_z = 2000
# Generation position:
x_gen_mm = 0
# Bulk wave speeds:
c_s_mpers = 3130
c_l_mpers = 6373
theta_crit_deg = np.rad2deg(np.arcsin(c_s_mpers / c_l_mpers))
# Leaky surface wave speed: (slightly slower than c_l)
c_lsaw_mpers = 6000

# Time instance to visualise:
# What radial distance would I like the direct bulk SV wavefront to have
# travelled?
r_mm = 7 * global_length_scale_factor
# Convert to a travel time using c_s:
t_snapshot_s = (r_mm * 10**-3) / c_s_mpers
# Plotting parameters:
c_map = cc.m_CET_D7
color_wavefronts = 'greenyellow'
lw_wavefronts = line_width_plots * 1.3

# Make a grid:
# Calculate pixel size:
pixel_size_mm = z_max_mm / n_pixels_z

# Build z-axis vector:
z_vector_mm = np.linspace(start=0, stop=z_max_mm,
                          num=n_pixels_z, endpoint=True)

# Build x-axis vector:
# Use of numpy arange() means x_end_mm will not be reached.  The highest x
# value will be the largest integer multiple of the step size away from
# x_start_mm that is less than x_end_mm.
x_vector_mm = np.arange(x_min_mm, x_max_mm, pixel_size_mm)

# Create 2D mesh grid in metres:
x_grid_m, z_grid_m = np.meshgrid(x_vector_mm * 10**-3, z_vector_mm * 10**-3)

# Compute the angle phi (bernstein coordinate system: phi is
# the angle of a ray from the surface) for all grid points:
phi_grid_deg = np.rad2deg(np.atan2(z_grid_m, x_grid_m))

# Bernstein and Spicer model XZ-plane displacement snapshot:
# Calculate wave slownesses from wave speeds:
s_l_sperm = 1 / c_l_mpers
s_t_sperm = 1 / c_s_mpers

(u_grid_m, v_grid_m
 ) = bernspice.get_u_v_displacement_components_at_iso_t_xz_plane_snapshot(
     x_grid_m, z_grid_m, t_snapshot_s, s_l_sperm, s_t_sperm
)

# Resolve u & v displacement components into different axes:
# Shear displacement in theta-direction in 2D circular polar coordinates
# centred on the source at (0,0).  A colormap of u_theta_m will
# highlight the direct shear wave:
u_theta_grid_m = bernspice.resolve_u_theta_from_bernstein_u_and_v(
    u_grid_m, v_grid_m, phi_grid_deg
)
# Radial dislacement in 2D circular polar coordinates.
# A colormap of this will highlight the longitudinal wave:
u_r_grid_m = bernspice.resolve_u_r_from_bernstein_u_and_v(
    u_grid_m, v_grid_m, phi_grid_deg
)
# Displacement in theta_crit direction.
# A colormap of this will highlight the shear head wave:
u_theta_crit_grid_m = bernspice.resolve_u_theta_crit_from_bernstein_u_and_v(
    u_grid_m, v_grid_m, c_l_mpers, c_s_mpers
)

im = plt.imshow(u_theta_crit_grid_m)
im.set_clim(-1e-8, 1e-8)

#%%

# Plotting calculations:

# Calculate the lateral distance travelled by a ray traversing the full
# thickness of the image (z_max_mm) at the critical angle:
dx_theta_b_mm = tan_theta_crit(c_l_mpers, c_s_mpers) * z_max_mm
x_coords_theta_crit_lines_mm = [(x_gen_mm - dx_theta_b_mm),
                                x_gen_mm,
                                (x_gen_mm + dx_theta_b_mm)]

# Calculations for rays:
# Direct:
theta_direct_ray_deg = 60
dx_direct_ray_tip_mm = np.sin(np.deg2rad(theta_direct_ray_deg)) * r_mm
x_direct_ray_tip_mm = x_gen_mm + dx_direct_ray_tip_mm
z_direct_ray_tip_mm = np.cos(np.deg2rad(theta_direct_ray_deg)) * r_mm
xz_direct_ray_tip_mm = (x_direct_ray_tip_mm, z_direct_ray_tip_mm)
# Head:
dx_LSAW_ray_mm = 4
x_LSAW_ray_mm = x_gen_mm + dx_LSAW_ray_mm
xz_LSAW_ray_tip_mm = (x_LSAW_ray_mm, 0)

t_birth_ray_s = (dx_LSAW_ray_mm * 10**-3) / c_lsaw_mpers
dt_as_SV_ray_s = t_snapshot_s - t_birth_ray_s
l_as_SV_ray_mm = (dt_as_SV_ray_s * c_s_mpers) * 1000
x_head_ray_tip_mm = x_LSAW_ray_mm + (np.sin(np.deg2rad(theta_crit_deg))
                                     * l_as_SV_ray_mm)
z_head_ray_tip_mm = np.cos(np.deg2rad(theta_crit_deg)) * l_as_SV_ray_mm
xz_head_ray_tip_mm = (x_head_ray_tip_mm, z_head_ray_tip_mm)


# Extent of images in x z:
extent_xz = [0, x_max_mm,
             z_max_mm, 0]
# Define colors:
color_direct = 'blue'
color_head = 'red'
c_map = cc.m_CET_D7
color_pos = c_map(1.0)
color_neg = c_map(0.0)

# Define plotting functions:


def xz_axes_format(ax, labeltop=False):
    ax.set_ylabel(r'$z$ (mm)', fontsize=fontsize_labels_pt)
    ax.yaxis.labelpad = -4
    ax.tick_params(top=True, labeltop=labeltop,
                   bottom=False, labelbottom=False)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax.xaxis.set_label_position('top')
    set_ax_ticks(ax)


def add_lines_at_theta_crit(ax, color='k'):
    ax.plot(x_coords_theta_crit_lines_mm, [z_max_mm, 0, z_max_mm],
            ls=my_dashes, color=color, lw=line_width_plots)


def add_line_normal(ax, color='k'):
    ax.axvline(x_gen_mm, ls=my_dashes, lw=line_width_plots, color=color)


def set_ax_ticks(ax):
    # x axis:
    ax.set_xticks([0, x_max_mm])

    # y axis:
    ax.set_yticks([0, 10])


def combined_SV_wavefront_sketch(ax):
    # Set ax aspect and limits:
    ax.set_ylim(z_max_mm, 0)
    ax.set_xlim(0, x_max_mm)
    ax.set_aspect('equal')
    # Add an annulus representing the direct wavefront:
    # Temporal half-width of Ricker waveform:
    t_half_width_patches_s = 0.1 * 10**-6
    radius_outer_mm = ((t_snapshot_s + t_half_width_patches_s)
                       * c_s_mpers) / 10**-3
    pulse_width_mm = (t_half_width_patches_s * 2 * c_s_mpers) / 10**-3
    annulus = patches.Annulus((x_gen_mm, 0), radius_outer_mm, pulse_width_mm,
                              lw=line_width_plots, ec=color_direct,
                              fc=to_rgba(color_direct, 0.3))
    ax.add_patch(annulus)

    # Add polygons and lines representing the head wave front:
    dx_LSAW_start_mm = ((t_snapshot_s - t_half_width_patches_s)
                        * c_lsaw_mpers) / 10**-3
    dx_LSAW_stop_mm = ((t_snapshot_s + t_half_width_patches_s)
                       * c_lsaw_mpers) / 10**-3
    radius_direct_start_mm = (
        (t_snapshot_s - t_half_width_patches_s) * c_s_mpers) / 10**-3
    radius_direct_stop_mm = (
        (t_snapshot_s + t_half_width_patches_s) * c_s_mpers) / 10**-3
    dx_direct_crit_start_mm = np.sin(np.deg2rad(
        theta_crit_deg)) * radius_direct_start_mm
    dx_direct_crit_stop_mm = np.sin(np.deg2rad(
        theta_crit_deg)) * radius_direct_stop_mm
    y_direct_crit_start_mm = np.cos(np.deg2rad(
        theta_crit_deg)) * radius_direct_start_mm
    y_direct_crit_stop_mm = np.cos(np.deg2rad(
        theta_crit_deg)) * radius_direct_stop_mm
    # Left:
    # Left transluscent polygon:
    alpha_H_polys = 0.3
    H_poly_left = patches.Polygon([[(x_gen_mm - dx_LSAW_start_mm), 0],
                                   [(x_gen_mm-dx_direct_crit_start_mm),
                                    y_direct_crit_start_mm],
                                   [(x_gen_mm-dx_direct_crit_stop_mm),
                                    y_direct_crit_stop_mm],
                                   [(x_gen_mm - dx_LSAW_stop_mm), 0]],
                                  ec='none',
                                  fc=to_rgba(color_head, alpha_H_polys))
    ax.add_patch(H_poly_left)
    # Left start line:
    ax.plot([(x_gen_mm - dx_LSAW_start_mm),
             (x_gen_mm-dx_direct_crit_start_mm)],
            [0, y_direct_crit_start_mm], color=color_head, lw=line_width_plots)
    # Left stop line:
    ax.plot([(x_gen_mm - dx_LSAW_stop_mm), (x_gen_mm-dx_direct_crit_stop_mm)],
            [0, y_direct_crit_stop_mm], color=color_head, lw=line_width_plots)
    # Right:
    # Right transluscent polygon:
    H_poly_right = patches.Polygon([[(x_gen_mm + dx_LSAW_start_mm), 0],
                                    [(x_gen_mm+dx_direct_crit_start_mm),
                                    y_direct_crit_start_mm],
                                    [(x_gen_mm+dx_direct_crit_stop_mm),
                                    y_direct_crit_stop_mm],
                                    [(x_gen_mm + dx_LSAW_stop_mm), 0]],
                                   ec='none',
                                   fc=to_rgba(color_head, alpha_H_polys))
    ax.add_patch(H_poly_right)
    # Right start line:
    ax.plot([(x_gen_mm + dx_LSAW_start_mm),
             (x_gen_mm+dx_direct_crit_start_mm)],
            [0, y_direct_crit_start_mm], color=color_head, lw=line_width_plots)
    # Right stop line:
    ax.plot([(x_gen_mm + dx_LSAW_stop_mm), (x_gen_mm+dx_direct_crit_stop_mm)],
            [0, y_direct_crit_stop_mm], color=color_head, lw=line_width_plots)

    # Add crit and normal lines:
    add_lines_at_theta_crit(ax, 'k')
    # add_line_normal(ax, 'k')
    # Label theta crit as an angle:
    add_angle_label(ax, 'k')

    # Text labelling Head wave front:
    ax.text(9.4, 4, 'SV head', color=color_head, fontsize=fontsize_labels_pt,
            weight='bold', rotation=theta_crit_deg)
    # Text labelling direct wave front:
    ax.text(3.4, 2.27, 'SV\ndirect', color=color_direct,
            fontsize=fontsize_labels_pt, weight='bold',
            ma='right')

    # Label 'Overlap' region just above theta crit:
    xy_arrow_head = (4.39, 5.36)
    xy_text = (6.68, 8)
    ax.annotate('Overlap region', xy_arrow_head, xy_text,
                arrowprops=dict_arrowprops_arrow, fontsize=fontsize_labels_pt)

    # Add a hybrid wavefront in a contrasting colour:
    # Form the hybrid wavefront from two sections: A sub-critical arc,
    # and a supercritical line.
    # Sub-critical arc
    diameter_hybrid_arc_mm = (radius_outer_mm - pulse_width_mm/2) * 2
    arc_hybrid = patches.Arc((x_gen_mm, 0), diameter_hybrid_arc_mm,
                             diameter_hybrid_arc_mm,
                             theta1=90 - theta_crit_deg, theta2=90,
                             lw=lw_wavefronts, color=color_wavefronts)
    ax.add_patch(arc_hybrid)
    # Super-critical line:
    x_surface_mm = x_gen_mm + (dx_LSAW_start_mm + dx_LSAW_stop_mm) / 2
    x_at_crit_mm = x_gen_mm + (dx_direct_crit_start_mm +
                               dx_direct_crit_stop_mm) / 2
    z_at_crit_mm = (y_direct_crit_start_mm + y_direct_crit_stop_mm) / 2
    ax.plot([x_surface_mm, x_at_crit_mm], [0, z_at_crit_mm],
            color=color_wavefronts, lw=lw_wavefronts, zorder=2)


def add_colorbar(fig, ax, im, visible=True):
    cbar_ticks = [-10, 0, 10]
    cbar = fig.colorbar(im, ax=ax, aspect=8, pad=0.02,
                        shrink=0.8)
    # cbar.set_label('Displacement (Arb.)', size=fontsize_labels_pt)
    cbar.ax.set_yticks(cbar_ticks,
                       labels=['-', '0', '+'],)
    cbar.ax.tick_params(labelsize=fontsize_labels_pt)
    cbar.ax.set_visible(visible)


def add_angle_label(ax, color='k', x_mm=x_gen_mm, left_side=False):
    diameter_angle_arc_mm = 6
    if left_side:
        arc = patches.Arc((x_mm, 0), diameter_angle_arc_mm,
                          diameter_angle_arc_mm,
                          theta1=90, theta2=90+theta_crit_deg,
                          lw=line_width_plots, color=color)
        ax.text(x_mm - 0.4, 4, r'$\theta$*',
                fontsize=fontsize_labels_pt, color=color, ha='right')
    else:
        arc = patches.Arc((x_mm, 0), diameter_angle_arc_mm,
                          diameter_angle_arc_mm,
                          theta1=90-theta_crit_deg, theta2=90,
                          lw=line_width_plots, color=color)
        ax.text(x_mm + 0.4, 4, r'$\theta$*',
                fontsize=fontsize_labels_pt, color=color)
    ax.add_patch(arc)


def axis_label(ax, label):
    ax.text(0.5, -0.2, label, fontsize=fontsize_labels_pt,
            transform=ax.transAxes, ha='center', va='center')


def displacement_key_direct(ax):
    # Add a circular arc aligned with the direct wavefront:
    arc = patches.Arc((x_gen_mm, 0), 20, 20, theta1=20, theta2=45,
                      lw=line_width_plots, color='k')
    ax.add_patch(arc)
    # Add arrows to show the directions of positive and negative
    # shear displacement:
    xy_midpoint = (9.47, 6)
    ax.annotate('', xy=(10.40, 4.1),
                xytext=xy_midpoint, xycoords='data',
                arrowprops={**dict_arrowprops_arrow,
                            'connectionstyle': 'arc3,rad=0.07',
                            'color': color_pos})
    ax.annotate('', xy=(8.12, 7.65),
                xytext=xy_midpoint, xycoords='data',
                arrowprops={**dict_arrowprops_arrow,
                            'connectionstyle': 'arc3,rad=-0.07',
                            'color': color_neg})
    # Add text labelling u_theta, + and -:
    ax.text(9.9, 6.8, r'$u_\theta$', color='k')
    ax.text(11, 5.1, '+', fontsize=fontsize_labels_pt, color=color_pos)
    ax.text(8.96, 8.2, '-', fontsize=fontsize_labels_pt, color=color_neg)


def displacement_key_head(ax):
    # Add a line aligned with the head wavefront:
    x_start = 5.92
    y_start = 7.15
    line_length = 5
    dx = np.cos(np.deg2rad(theta_crit_deg)) * line_length
    dy = np.sin(np.deg2rad(theta_crit_deg)) * line_length
    x_end = x_start + dx
    y_end = y_start - dy
    ax.plot([x_start, x_end], [y_start, y_end], color='k',
            lw=line_width_plots)
    # Add positive and negative displacement arrows:
    x_mid = 8.62
    y_mid = 6.89
    arrow_length = 2
    dx_arrow = np.cos(np.deg2rad(theta_crit_deg)) * arrow_length
    dy_arrow = np.sin(np.deg2rad(theta_crit_deg)) * arrow_length
    x_right = x_mid + dx_arrow
    x_left = x_mid - dx_arrow
    y_right = y_mid - dy_arrow
    y_left = y_mid + dy_arrow
    ax.annotate('', xy=(x_right, y_right),
                xytext=(x_mid, y_mid), xycoords='data',
                arrowprops={**dict_arrowprops_arrow,
                            'color': color_pos})
    ax.annotate('', xy=(x_left, y_left),
                xytext=(x_mid, y_mid), xycoords='data',
                arrowprops={**dict_arrowprops_arrow,
                            'color': color_neg})
    # Add text:
    ax.text(8.61, 8.42, r'$u_{\theta*}$', color='k')
    ax.text(10.14, 7.29, '+', fontsize=fontsize_labels_pt, color=color_pos)
    ax.text(7.27, 8.76, '-', fontsize=fontsize_labels_pt, color=color_neg)


def top_left_label(ax, label_string, color):
    # Place a white text box with a black line border in the top
    # left corner of ax:
    props = dict(boxstyle='square', facecolor='k', edgecolor='k',
                 lw=line_width_plots)
    ax.text(0.01, 0.97, label_string, transform=ax.transAxes,
            fontsize=fontsize_labels_pt, verticalalignment='top',
            bbox=props, color=color)


def axis_title(ax, title_string, color):
    ax.set_title(title_string, fontsize=fontsize_labels_pt,
                 weight='bold')


def plot_u_theta_snapshot(ax):
    # Plot displacement colormap:
    xz_axes_format(ax, True)
    # Find max(abs()) value for colormap limits:
    # max_mod_displacement_m = np.max(np.abs(u_theta_grid_m))
    clim_symmetric_nm = 10
    # Imshow:
    im_direct = ax.imshow(u_theta_grid_m / 1e-9, cmap=c_map,
                          vmin=-clim_symmetric_nm,
                          vmax=clim_symmetric_nm,
                          extent=extent_xz)
    add_lines_at_theta_crit(ax)
    add_colorbar(fig, ax, im_direct)
    # add_line_normal(ax)
    add_angle_label(ax)
    displacement_key_direct(ax)

    # Plot a ray:
    ax.annotate('', xy=xz_direct_ray_tip_mm,
                xytext=(x_gen_mm, 0), xycoords='data',
                arrowprops={**dict_arrowprops_arrow,
                            'color': 'k'})

    return im_direct


def plot_u_theta_crit_snapshot(ax):
    # Plot displacement colormap:
    xz_axes_format(ax)
    # max_mod_displacement_m = np.max(np.abs(u_theta_crit_grid_m))
    c_lim_symmetric_nm = 10
    im_head = ax.imshow(u_theta_crit_grid_m / 1e-9, cmap=c_map,
                        vmin=-c_lim_symmetric_nm,
                        vmax=c_lim_symmetric_nm,
                        extent=extent_xz)
    add_lines_at_theta_crit(ax)
    add_colorbar(fig, ax, im_head)
    # add_line_normal(ax2)
    add_angle_label(ax)
    displacement_key_head(ax)

    # Plot a ray:
    # SV leg:
    ax.annotate('', xy=xz_head_ray_tip_mm,
                xytext=xz_LSAW_ray_tip_mm, xycoords='data',
                arrowprops={**dict_arrowprops_arrow,
                            'color': 'k'})
    # LSAW leg:
    color_lsaw = [1, 0, 1]
    ax.annotate('', xy=xz_LSAW_ray_tip_mm,
                xytext=(x_gen_mm, 0), xycoords='data',
                clip_on=False,
                arrowprops={**dict_arrowprops_arrow,
                            'color': color_lsaw})
    # LSAW text label:
    ax.text(x_gen_mm + (dx_LSAW_ray_mm/2) + 0.2, -0.2, 'LSAW',
            fontsize=fontsize_labels_pt, ha='center', va='bottom',
            color=color_lsaw, clip_on=False)
    # Short surface normal:
    ax.plot([x_LSAW_ray_mm, x_LSAW_ray_mm], [0, 3], ls=my_dashes,
            lw=line_width_plots, color='k')
    # Annotate angle:
    add_angle_label(ax, x_mm=x_LSAW_ray_mm)


# # Test plotting of complex phase:
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout='constrained')

# ax1.imshow(np.real(u_theta_grid_direct_S_nm))
# ax2.imshow(np.imag(u_theta_grid_direct_S_nm))
# ax3.imshow(np.angle(u_theta_grid_direct_S_nm))

# matplotlib.use('TkAgg')

# Plotting figure:
set_mpl_fonts_Times_New_Roman()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                    figsize=(width_fig_one_column_jasa_in, 5),
                                    layout='constrained', sharex=True)

# Direct SV displacements:
im_direct = plot_u_theta_snapshot(ax1)
axis_label(ax1, '(a)\n')
ax1.set_xlabel(r'$x$ (mm)', fontsize=fontsize_labels_pt)
ax1.xaxis.labelpad = -5

# Head SV displacements:
plot_u_theta_crit_snapshot(ax2)
axis_label(ax2, '(b)\n')

# Combined SV wavefront sketch:
xz_axes_format(ax3)
combined_SV_wavefront_sketch(ax3)
axis_label(ax3, '(c)\n')

# Add an invisible colorbar to ax3 to align the three plots horizontally:
add_colorbar(fig, ax3, im_direct, False)

# Label top colorbar:
# fig.text(0.78, 0.635, 'Displacement\n(Arb.)', fontsize=fontsize_labels_pt,
#          ma='right')

# Place labels above colorbars:
xfig_cbar_labels = 0.865
fontsize_boost = 2
fig.text(xfig_cbar_labels, 0.955, r'$u_{\theta}$',
         fontsize=fontsize_labels_pt + fontsize_boost,
         ma='center')
fig.text(xfig_cbar_labels, 0.62, r'$u_{\theta*}$',
         fontsize=fontsize_labels_pt + fontsize_boost,
         ma='center')

plt.show()

# %%

# Saving:
# fig.savefig(r'Figure folders\Fig SV wavefronts\SV_wavefronts_BernSpice_JASA.pdf',
#             format='pdf', bbox_inches='tight', dpi=600)
# fig.savefig(r'Figure folders\Fig SV wavefronts\SV_wavefronts_BernSpice_JASA.png',
#             format='png', dpi=200, bbox_inches='tight')

# %%
