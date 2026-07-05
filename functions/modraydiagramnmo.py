import numpy as np
from matplotlib import patches

from corevariables.modieeeplotting import (line_width_plots,
                                           fontsize_labels_pt,
                                           lw_lasers,
                                           lw_boundaries,
                                           dict_arrowprops_line,
                                           dict_arrowprops_arrow,
                                           my_dashes)


def plot_ray_diagram_nmo(ax, x_max_mm, b_mm):
    # Arrows:
    x_gen_mm = 6
    x_det_mm = 0
    x_midpoint_mm = (x_gen_mm - x_det_mm) / 2
    x_send_head = x_gen_mm - (x_midpoint_mm)/2
    x_receive_head = (x_midpoint_mm)/2
    xy_send_head = (x_send_head, b_mm/2)
    xy_receive_head = (x_receive_head, b_mm/2)
    theta_GRD = np.rad2deg(np.atan2(x_midpoint_mm, b_mm))
    xy_reflec = (x_midpoint_mm, b_mm)
    xy_gen = (x_gen_mm, 0)
    xy_det = (x_det_mm, 0)

    # Other parameters:
    laser_line_length = 1.5
    diameter_angle_arc = 1.7

    # Format original axis:
    ax.set_aspect('equal')
    # The x-axis will span from 0 to x_max:
    ax.set_xlim(0, x_max_mm)
    ax.autoscale(enable=False)
    # Hide the axes spines and ticks:
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    # Spines invisible, but tick marks are still there:
    ax.spines['top'].set_position('zero')
    ax.tick_params(top=True, labeltop=True,
                   bottom=False, labelbottom=False)
    ax.set_xticks([10])
    ax.set_yticks([0, 2])
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax.set_ylabel('Depth $z$ (mm)', fontsize=fontsize_labels_pt)
    # Add a grey dotted line showing the highest-offset ray-path:
    my_grey = 'k'  # [0.7, 0.7, 0.7]
    ax.text(x_max_mm-0.3, -0.5, r'$x_\mathrm{Max}=$ ', fontsize=fontsize_labels_pt,
            ha='right', color=my_grey)
    ax.plot([0, x_max_mm/2, x_max_mm],
            [0, b_mm, 0], ls=':', color=my_grey, lw=line_width_plots)
    # Add a vline and an angle arc marking theta_max = 68 deg:
    theta_max_deg = np.rad2deg(np.atan2(x_max_mm/2, b_mm))
    ax.vlines(x_max_mm, 0, b_mm, color=my_grey, linewidth=0.7,
              linestyles=(0, (4, 4)), clip_on=False)
    arc = patches.Arc((x_max_mm, 0), diameter_angle_arc, diameter_angle_arc,
                      theta1=90, theta2=90 + theta_max_deg,
                      lw=0.7, color=my_grey)
    ax.add_patch(arc)
    # Add text theta max:
    ax.text(x_max_mm - 1.3, 0.63, r'$\theta_\mathrm{Max}$',
            fontsize=fontsize_labels_pt, va='top', color=my_grey)
    ax.text(x_max_mm - 1.8, 1.4, r'$=68\degree$',
            fontsize=fontsize_labels_pt, va='top', color=my_grey)
    # Add a dashed line for the surface normal:
    ax.vlines(x_gen_mm, 0, b_mm, 'k', linewidth=0.7, linestyles=(0, (4, 4)))
    # Add red laser:
    ax.vlines(x_gen_mm, 0, -laser_line_length, 'red', linewidth=lw_lasers)
    # Add green laser:
    ax.vlines(x_det_mm, 0, -laser_line_length, 'lime',
              linewidth=lw_lasers, clip_on=False)
    # Add a doublearrow marking the offset, x:
    y_doublearrow = -1.2
    ax.annotate('', xy=(x_det_mm, y_doublearrow),
                xytext=(x_gen_mm, y_doublearrow), xycoords='data',
                arrowprops=dict(arrowstyle='<->', shrinkA=0.5, shrinkB=0.5,
                                lw=line_width_plots), clip_on=False)
    ax.text(x_midpoint_mm, -1.3, r'$x$', ha='center', va='bottom',
            fontsize=fontsize_labels_pt, color='k', clip_on=False)
    # Add a dashed line for the surface normal under the detection laser:
    ax.vlines(x_det_mm, 0, b_mm, 'k', clip_on=False,
              linewidth=0.7, linestyles=my_dashes)
    # Add a dashed line for the surface normal at reflection:
    ax.vlines(x_midpoint_mm, 0, b_mm, 'k', linewidth=0.7, linestyles=my_dashes)
    # Add send half-arrow:
    ax.annotate('', xy=xy_send_head,
                xytext=xy_gen, xycoords='data',
                arrowprops=dict_arrowprops_arrow)
    # Add send half line:
    ax.annotate('', xy=xy_reflec,
                xytext=xy_gen, xycoords='data',
                arrowprops=dict_arrowprops_line)
    # Add receive half-arrow:
    ax.annotate('', xy=xy_receive_head,
                xytext=xy_reflec, xycoords='data',
                arrowprops=dict_arrowprops_arrow)
    # Add receive half-line:
    ax.annotate('', xy=xy_det,
                xytext=xy_reflec, xycoords='data',
                arrowprops=dict_arrowprops_line)
    # Annotate angle theta:
    # At generation:
    arc = patches.Arc(xy_gen, diameter_angle_arc, diameter_angle_arc,
                      theta1=90, theta2=90 + theta_GRD,
                      lw=0.7, color='k')
    ax.add_patch(arc)
    # Add text theta:
    ax.text(x_gen_mm - 0.75, 0.95, r'$\theta$', fontsize=fontsize_labels_pt,
            va='top')
    # At reflection:
    arc = patches.Arc(xy_reflec, diameter_angle_arc, diameter_angle_arc,
                      theta1=-90, theta2=-90 + theta_GRD,
                      lw=0.7, color='k')
    ax.add_patch(arc)
    # Add text theta:
    ax.text(x_midpoint_mm + 0.75, 1.1, r'$\theta$',
            fontsize=fontsize_labels_pt,
            ha='right', va='bottom')
    # At detection:
    arc = patches.Arc(xy_det, diameter_angle_arc, diameter_angle_arc,
                      theta1=90 - theta_GRD, theta2=90,
                      lw=0.7, color='k')
    ax.add_patch(arc)
    # Add text theta:
    ax.text(x_det_mm + 0.35, 0.95, r'$\theta$', fontsize=fontsize_labels_pt,
            ha='left', va='top')
    # Add a little arrow to show the surface normal displacement:
    ax.annotate('', xy=(x_det_mm + 0.3, -0.9), xytext=(x_det_mm + 0.3, -0.2),
                xycoords='data', arrowprops=dict_arrowprops_arrow)
    # add text labelling u_z:
    ax.text(x_det_mm + 0.6, -0.5, r'$u_\mathrm{z}$',
            fontsize=fontsize_labels_pt, ha='left')

    ax.set_ylim(b_mm + 1, -laser_line_length)
    x_lims = [0, x_max_mm]
    ax.set_xlim(x_lims)

    # Plot lines for the surfaces:
    ax.plot(x_lims, [0, 0], 'k', lw=lw_boundaries)
    ax.plot(x_lims, [b_mm, b_mm], 'k', lw=lw_boundaries)
