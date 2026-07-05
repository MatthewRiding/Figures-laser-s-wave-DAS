from matplotlib import patches
import numpy as np

from corevariables.modieeeplotting import (line_width_plots,
                                           dict_arrowprops_arrow,
                                           dict_arrowprops_line,
                                           fontsize_labels_pt,
                                           lw_lasers,
                                           lw_boundaries,
                                           my_dashes)


# Ray diagram parameters:
x_max = 20
x_lasers = 5
laser_line_length = 1.8
laser_half_separation = 0.13

diameter_angle_arc = 5
xz_gen = (x_lasers, 0)
x_sdh = 9.75
z_sdh = 5
r_sdh = 0.5
theta_rad = np.atan2((x_sdh - x_lasers),
                     z_sdh)

x_reflec = x_sdh - (r_sdh * np.sin(theta_rad))
z_reflec = z_sdh - (r_sdh * np.cos(theta_rad))
xz_reflec = (x_reflec, z_reflec)

x_midpoint = x_lasers + (x_reflec - x_lasers)/2
z_midpoint = (z_sdh - r_sdh) / 2
xz_midpoint = (x_midpoint, z_midpoint)


xz_sdh = (x_sdh, z_sdh)

z_max = 6.5


def plot_ray_diagram_sdh_mss(ax, axis_label_string):
    # Format original axis:
    ax.set_aspect('equal')
    # The y-axis will span from 0-1, and the x-axis from 0 to x_max.
    ax.set_xlim(0, x_max)
    ax.autoscale(enable=False)
    # Hide the axes spines and ticks:
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax.tick_params(top=True, labeltop=True,
                   bottom=False, labelbottom=False)
    ax.spines['top'].set_position('zero')
    ax.set_xticks([0, x_sdh, 20])
    ax.set_xticklabels(['0', r'$x_\mathrm{SDH}\,=$' + '\n' + r'$9.75$', '20'])
    ax.set_yticks([0, 5])
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax.set_ylabel('Depth $z$ (mm)', fontsize=fontsize_labels_pt)
    # Add a letter label for the figure:
    ax.set_xlabel(axis_label_string, fontsize=fontsize_labels_pt, labelpad=8)
    # Draw rect representing solid sample:
    rect = patches.Rectangle((0, 0), x_max, z_max, linewidth=lw_boundaries,
                             clip_on=False, edgecolor='k', facecolor='none',
                             zorder=2)
    # Add a dashed line for the surface normal:
    ax.vlines(x_lasers, 0, z_max, 'k', linewidth=0.7, linestyles=my_dashes)

    # Add red laser:
    ax.vlines(x_lasers-laser_half_separation, 0, -laser_line_length, 'red',
              linewidth=lw_lasers)
    # Add detection laser:
    ax.vlines(x_lasers+laser_half_separation, 0, -laser_line_length, 'lime',
              linewidth=lw_lasers)
    # Add send arrow:
    ax.annotate('', xy=xz_reflec,
                xytext=xz_gen, xycoords='data',
                arrowprops=dict_arrowprops_arrow)
    # Add return arrow:
    ax.annotate('', xy=xz_midpoint,
                xytext=xz_reflec, xycoords='data',
                arrowprops=dict_arrowprops_arrow)
    # Annotate angle theta:
    arc = patches.Arc(xz_gen, diameter_angle_arc, diameter_angle_arc,
                      theta1=90 - np.rad2deg(theta_rad), theta2=90,
                      lw=0.7, color='k')
    ax.add_patch(arc)
    # Add text theta:
    ax.text(6, 4, r'$\theta$', fontsize=fontsize_labels_pt)
    # Add circle representing SDH:
    circ = patches.Circle(xz_sdh, r_sdh, edgecolor='k', facecolor='none',
                          lw=lw_boundaries)
    ax.add_patch(circ)
    # Add text labelling z_sdh:
    ax.text(x_sdh + 1, z_sdh, r'$r_\mathrm{SDH}\,=0.5$mm',
            fontsize=fontsize_labels_pt, ha='left', va='center')
    # Add a dashed line marking x_SDH:
    ax.vlines(x_sdh, 0, z_sdh, 'k', lw=0.7, linestyles=my_dashes)

    # Add a little arrow to show the surface normal displacement:
    ax.annotate('', xy=(x_lasers-0.8, -laser_line_length+0.25),
                xytext=(x_lasers-0.8, -0.25),
                xycoords='data', arrowprops=dict_arrowprops_arrow)
    # add text labelling u_z:
    ax.text(x_lasers - 1.5, -(laser_line_length/2), r'$u_\mathrm{z}$',
            fontsize=fontsize_labels_pt, ha='right', va='center')
    ax.set_ylim(z_max, -laser_line_length)
    ax.set_xlim(0, x_max)

    # Add magenta array line:
    ax.hlines(-0.2, 0, 20, color='magenta', lw=1, ls=':')

    # Add rectangle last so that it is in front:
    ax.add_patch(rect)
