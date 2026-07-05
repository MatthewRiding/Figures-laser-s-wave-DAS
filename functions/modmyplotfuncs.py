import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from scipy.signal import hilbert
import colorcet as cc

from corevariables.modieeeplotting import (fontsize_labels_pt,
                                           line_width_plots,
                                           my_dashes, color_phase,
                                           color_abs,
                                           color_vcf,
                                           lw_lasers,
                                           lw_boundaries,
                                           dict_arrowprops_arrow,
                                           dict_arrowprops_line)


def theta_line(ax, x_theta_mm, label):
    # Plot a dashed axvline at the x value corresponding to the given angle:
    ax.axvline(x_theta_mm, color=(1, 1, 1), dashes=(4, 4), linewidth=0.7)
    t_lims = ax.get_ylim()
    t_text = t_lims[1] + ((t_lims[0] - t_lims[1]) * 0.96)
    ax.text(x_theta_mm, t_text, label, color=(1, 1, 1),
            fontsize=fontsize_labels_pt)


def axis_label(ax, label):
    ax.set_title(label, fontsize=fontsize_labels_pt, y=0, pad=-9)


def common_angle_plot_formatting(ax, theta_crit_deg):
    ax.set_xlim(0, 90)
    ax.axvline(theta_crit_deg, ls=my_dashes, lw=line_width_plots, color='k')
    ax.axvline(45, ls=my_dashes, lw=line_width_plots, color='k')
    ax.axhline(color='k', lw=0.8)
    ax.set_xticks([0, 30, 60, 90], minor=False)
    ax.set_xticks(np.linspace(10, 80, 8), minor=True)
    ax.set_yticks(np.linspace(-3, 3, 7), minor=True)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)


def real_and_imag_plots(ax, y_complex, x):
    real_handle, = ax.plot(x, np.real(y_complex), label='Real',
                           color='k', lw=line_width_plots)
    imag_handle, = ax.plot(x, np.imag(y_complex), label='Imag',
                           color='b', lw=line_width_plots)
    return real_handle, imag_handle


def abs_plot(ax, y_complex, x):
    abs_handle, = ax.plot(x, np.abs(y_complex),
                          label='Abs', color=color_abs, zorder=-1,
                          lw=line_width_plots)
    return abs_handle


def phase_plot(ax, y_complex, x, yyaxis=True):
    if yyaxis:
        ax_phase = ax.twinx()
    else:
        ax_phase = ax

    phase_handle, = ax_phase.plot(x, np.rad2deg(np.angle(y_complex)),
                                  color=color_phase, label='Phase',
                                  lw=line_width_plots, ls='dotted')
    if yyaxis:
        ax_phase.tick_params(axis='y', labelcolor=color_phase)
        ax_phase.tick_params(axis='y', which='major', labelsize=8, pad=1)
        ax_phase.set_ylim(-180, 180)
        ax_phase.set_yticks([-180, -90, 0, 90, 180])
    return phase_handle, ax_phase


def invisible_plots_for_legend(ax):
    real_handle, imag_handle = real_and_imag_plots(ax, -90, x=-90)
    abs_handle = abs_plot(ax, -90, x=-90)
    phase_handle, ax_phase = phase_plot(ax, -360, -90, yyaxis=False)
    return real_handle, imag_handle, abs_handle, phase_handle


def plot_inset_ray_diagram(ax, grd='g'):
    # Plot a small inset axes box in the bottom left corner of ax.
    # For use on each of the axes in the figure showing the line
    # plots of G_SV, R_SV-SV and D_SV.
    inset_bounds = [0, 0, 0.18, 0.4]
    ax_inset = ax.inset_axes(inset_bounds)
    ax_inset.set_aspect('equal')
    # The y-axis will span from 0-1, and the x-axis from 0 to x_max.
    x_max = 0.65
    ax_inset.autoscale(enable=False)
    # Hide the axes ticks:
    ax_inset.tick_params(top=False, labeltop=False,
                         left=False, labelleft=False,
                         bottom=False, labelbottom=False)

    # Set constant params:
    y_inset_top_surface = 0.8
    y_inset_bottom_surface = 0.2
    y_inset_normal_end = 0.1
    x_laser_offset = 0.2
    diameter_arc = 0.7
    # Set arrow length via x and y spans:
    dx_arrow_gd = 0.57
    dy_arrow_gd = 0.47
    theta_deg_gd = np.rad2deg(np.atan2(dx_arrow_gd, dy_arrow_gd))
    dx_arrow_r = 0.5
    dy_arrow_r = 0.4
    theta_deg_r = np.rad2deg(np.atan2(dx_arrow_r, dy_arrow_r))

    def plot_surface(y_inset_surface=y_inset_top_surface):
        ax_inset.axhline(y_inset_surface, lw=lw_boundaries,
                         color='k')

    def plot_normal(x_normal=-x_laser_offset,
                    y_inset_normal_top=y_inset_top_surface,
                    y_inset_normal_bottom=y_inset_bottom_surface):
        ax_inset.axvline(x_normal, y_inset_normal_bottom,
                         y_inset_normal_top, ls='--', lw=line_width_plots,
                         color='k')

    def plot_laser(laser_color='red', x_laser=-x_laser_offset):
        ax_inset.plot([x_laser, x_laser], [y_inset_top_surface, 1],
                      lw=lw_lasers, color=laser_color)

    def plot_arrow(direction='se'):
        # Set coordinates of head and tail points:
        if direction == 'se':
            x_tail = - x_laser_offset
            y_tail = y_inset_top_surface
            x_head = x_tail + dx_arrow_gd
            y_head = y_tail - dy_arrow_gd
        elif direction == 'ne':
            x_head = x_laser_offset
            y_head = y_inset_top_surface
            x_tail = x_head - dx_arrow_gd
            y_tail = y_head - dy_arrow_gd

        xy_head = [x_head, y_head]
        xy_tail = [x_tail, y_tail]
        ax_inset.annotate('', xy_head, xy_tail,
                          arrowprops=dict_arrowprops_arrow)

    def plot_reflection_rays():
        # Plot pre-reflection ray:
        xy_head = [0, y_inset_bottom_surface]
        xy_tail = [0 - dx_arrow_r, y_inset_bottom_surface + dy_arrow_r]
        ax_inset.annotate('', xy_head, xy_tail,
                          arrowprops=dict_arrowprops_line)
        # Plot post-reflection arrow:
        xy_head = [0 + dx_arrow_r, y_inset_bottom_surface + dy_arrow_r]
        xy_tail = [0, y_inset_bottom_surface]
        ax_inset.annotate('', xy_head, xy_tail,
                          arrowprops=dict_arrowprops_arrow)

    def add_arc(sector='se'):
        # Set angular extents & location of theta character:
        if sector == 'se':
            xy_centre = [-x_laser_offset, y_inset_top_surface]
            theta_start = -90
            theta_stop = -90 + theta_deg_gd
            x_character = 0
            y_character = 0.15
        elif sector == 'sw':
            xy_centre = [x_laser_offset, y_inset_top_surface]
            theta_start = -90 - theta_deg_gd
            theta_stop = -90
            x_character = -0.05
            y_character = 0.15
        elif sector == 'nw':
            xy_centre = [0, y_inset_bottom_surface]
            theta_start = 90
            theta_stop = 90 + theta_deg_r
            x_character = -0.23
            y_character = 0.6

        # Annotate angle theta:
        arc = patches.Arc(xy_centre, diameter_arc, diameter_arc,
                          theta1=theta_start, theta2=theta_stop,
                          lw=0.7, color='k')
        ax_inset.add_patch(arc)

        # Add text theta:
        ax_inset.text(x_character, y_character, r'$\theta$',
                      fontsize=fontsize_labels_pt, ha='center')

    if grd == 'g':
        # Call functions using defaults:
        plot_laser()
        plot_normal(y_inset_normal_bottom=y_inset_normal_end)
        plot_arrow()
        add_arc()
        plot_surface()
    elif grd == 'd':
        x_d = x_laser_offset
        plot_laser('lime', x_d)
        plot_normal(x_d, y_inset_normal_bottom=y_inset_normal_end)
        plot_arrow('ne')
        add_arc('sw')
        plot_surface()
    elif grd == 'r':
        plot_normal(0, y_inset_normal_top=0.9)
        add_arc('nw')
        plot_reflection_rays()
        plot_surface(y_inset_bottom_surface)

    ax_inset.set_ylim(0, 1)
    ax_inset.set_xlim(-x_max, x_max)


def plot_inset_ray_diagram_combined(ax, grd):
    # Set constant params:
    y_top_surface = 0.8
    y_bottom_surface = 0.1
    diameter_arc = 0.7

    def make_inset_ax(inset_bounds):
        # Plot an inset axes box in the bottom left corner of ax.
        # For use on each of the axes in the figure showing the line
        # plots of G_SV, R_SV-SV and D_SV.
        ax_inset = ax.inset_axes(inset_bounds)
        ax_inset.set_aspect('equal')
        ax_inset.autoscale(enable=False)
        # Hide the axes ticks:
        ax_inset.tick_params(top=False, labeltop=False,
                             left=False, labelleft=False,
                             bottom=False, labelbottom=False)
        return ax_inset

    def plot_surface(ax_inset, y_inset_surface=y_top_surface):
        ax_inset.axhline(y_inset_surface, lw=lw_boundaries,
                         color='k')

    def plot_normal(ax_inset, x_normal,
                    y_inset_normal_top=y_top_surface,
                    y_inset_normal_bottom=y_bottom_surface):
        ax_inset.axvline(x_normal, y_inset_normal_bottom,
                         y_inset_normal_top, ls='--', lw=line_width_plots,
                         color='k')

    def plot_laser(ax_inset, laser_color, x_laser):
        ax_inset.plot([x_laser, x_laser], [y_top_surface, 1],
                      lw=lw_lasers, color=laser_color)

    def plot_reflection_rays(ax_inset, x_laser_offset):
        # Plot pre-reflection ray:
        xy_head = [0, y_bottom_surface]
        xy_tail = [-x_laser_offset, y_top_surface]
        ax_inset.annotate('', xy_head, xy_tail,
                          arrowprops=dict_arrowprops_line)
        # Plot post-reflection arrow:
        xy_head = [x_laser_offset, y_top_surface]
        xy_tail = [0, y_bottom_surface]
        ax_inset.annotate('', xy_head, xy_tail,
                          arrowprops=dict_arrowprops_arrow)

    def add_arc(ax_inset, grd, x_laser_offset):
        x_g = - x_laser_offset
        dx_arrow_gr = x_laser_offset
        dy_arrow_gr = y_top_surface - y_bottom_surface
        theta_deg_gr = np.rad2deg(np.atan2(dx_arrow_gr, dy_arrow_gr))
        # Set angular extents & location of theta character:
        if grd == 'g':
            xy_centre = [-x_laser_offset, y_top_surface]
            theta_start = -90
            theta_stop = -90 + theta_deg_gr
            x_character = x_g + 0.24
            y_character = 0.17
        elif grd == 'd':
            xy_centre = [x_laser_offset, y_top_surface]
            theta_start = -90 - theta_deg_gr
            theta_stop = -90
            x_character = x_laser_offset - 0.28
            y_character = 0.17
        elif grd == 'r':
            xy_centre = [0, y_bottom_surface]
            theta_start = 90
            theta_stop = 90 + theta_deg_gr
            x_character = -0.25
            y_character = 0.46

        # Annotate angle theta:
        arc = patches.Arc(xy_centre, diameter_arc, diameter_arc,
                          theta1=theta_start, theta2=theta_stop,
                          lw=0.7, color='k')
        ax_inset.add_patch(arc)

        # Add text theta:
        ax_inset.text(x_character, y_character, r'$\theta$',
                      fontsize=fontsize_labels_pt, ha='center')

    if grd == 'gd':
        inset_bounds = [0.0117, 0, 0.24, 0.45]
        x_max = 0.9
        ax_inset = make_inset_ax(inset_bounds)
        x_g = -0.6
        # Plot lasers:
        plot_laser(ax_inset, 'red', x_g - 0.025)
        plot_laser(ax_inset, 'lime', x_g + 0.025)
        # Arrows: G and G*D:
        xy_gen = (x_g, y_top_surface)
        x_reflec = 0.3
        y_reflec = 0.3
        xy_reflec = (x_reflec, y_reflec)
        m_ = (y_reflec - y_top_surface)/(x_reflec - x_g)
        c_ = y_reflec - (m_ * x_reflec)
        theta_rad = np.atan2(1, np.abs(m_))
        x_midpoint = x_g + (x_reflec - x_g)/2
        y_midpoint = (m_ * x_midpoint) + c_
        xy_midpoint = (x_midpoint, y_midpoint)
        r_sdh = 0.1
        x_sdh = x_reflec + (r_sdh * np.sin(theta_rad))
        y_sdh = y_reflec - (r_sdh * np.cos(theta_rad))
        xy_sdh = (x_sdh, y_sdh)
        # Plot arrows:
        # Add send arrow:
        ax_inset.annotate('', xy=xy_reflec,
                          xytext=xy_gen, xycoords='data',
                          arrowprops=dict_arrowprops_arrow)
        # Add return arrow:
        ax_inset.annotate('', xy=xy_midpoint,
                          xytext=xy_reflec, xycoords='data',
                          arrowprops=dict_arrowprops_arrow)
        # Annotate angle theta:
        arc = patches.Arc(xy_gen, diameter_arc, diameter_arc,
                          theta1=-90, theta2=-90 + np.rad2deg(theta_rad),
                          lw=0.7, color='k')
        ax_inset.add_patch(arc)
        # Add text theta:
        ax_inset.text(x_g + 0.2, 0.15, r'$\theta$',
                      fontsize=fontsize_labels_pt)
        # Add circle representing SDH:
        circ = patches.Circle(xy_sdh, r_sdh, edgecolor='k', facecolor='none',
                              lw=lw_boundaries)
        ax_inset.add_patch(circ)

        # Plot normal:
        plot_normal(ax_inset, x_g)

        # Plot top surface:
        plot_surface(ax_inset)

    elif grd == 'grd':
        inset_bounds = [0.0117, 0, 0.28, 0.45]
        ax_inset = make_inset_ax(inset_bounds)
        # The y-axis will span from 0-1, and the x-axis from 0 to x_max.
        x_max = 1.08
        x_laser_offset = 0.9
        # Plot lasers:
        plot_laser(ax_inset, 'red', -x_laser_offset)
        plot_laser(ax_inset, 'lime', x_laser_offset)
        # Plot dashed normals:
        plot_normal(ax_inset, -x_laser_offset)
        plot_normal(ax_inset, 0)
        plot_normal(ax_inset, x_laser_offset)
        # Plot arrows:
        plot_reflection_rays(ax_inset, x_laser_offset)
        # Label angles:
        add_arc(ax_inset, 'g', x_laser_offset)
        add_arc(ax_inset, 'r', x_laser_offset)
        add_arc(ax_inset, 'd', x_laser_offset)
        # Plot top and bottom surfaces:
        plot_surface(ax_inset, y_top_surface)
        plot_surface(ax_inset, y_bottom_surface)

    ax_inset.set_ylim(0, 1)
    ax_inset.set_xlim(-x_max, x_max)


def G_SV_plot(ax, G_SV, theta_array_deg, theta_crit_deg):
    real_and_imag_plots(ax, G_SV, theta_array_deg)
    abs_plot(ax, G_SV, theta_array_deg)
    ax.set_ylabel('Generation\ndirectivity ' r'$G_\mathrm{SV}$',
                  fontsize=fontsize_labels_pt)
    common_angle_plot_formatting(ax, theta_crit_deg)
    ax.set_yticks([-3, 0, 3])
    phase_plot(ax, G_SV, theta_array_deg)


def R_SV_SV_plot(ax, R_SV_SV, theta_array_deg, theta_crit_deg,
                 label_phase=True):
    real_and_imag_plots(ax, R_SV_SV, theta_array_deg)
    abs_plot(ax, R_SV_SV, theta_array_deg)
    ax.set_ylabel('Reflection\ncoefficient ' r'$R_\mathrm{SV-SV}$',
                  fontsize=fontsize_labels_pt)
    common_angle_plot_formatting(ax, theta_crit_deg)
    handle, ax_phase = phase_plot(ax, R_SV_SV, theta_array_deg)
    if label_phase:
        ax_phase.set_ylabel('Phase angle (°)', color=color_phase,
                            fontsize=fontsize_labels_pt, x=1)
    ax.set_ylim(-1.1, 1.1)


def R_SV_L_plot(ax, R_SV_L, theta_array_deg, theta_crit_deg):
    real_and_imag_plots(ax, R_SV_L, theta_array_deg)
    abs_plot(ax, R_SV_L, theta_array_deg)
    ax.set_ylabel('Reflection\ncoefficient ' r'$R_\mathrm{SV-L}$',
                  fontsize=fontsize_labels_pt)
    common_angle_plot_formatting(ax, theta_crit_deg)
    phase_plot(ax, R_SV_L, theta_array_deg)


def theta_l_plot(ax, theta_l_rad_complex, theta_array_deg, theta_crit_deg):
    ax.plot(theta_array_deg, np.rad2deg(np.real(theta_l_rad_complex)),
            label='Real', color='k', lw=line_width_plots)
    ax.plot(theta_array_deg, np.rad2deg(np.imag(theta_l_rad_complex)),
            label='Imag', color='b', lw=line_width_plots)
    ax.set_ylabel('Angle of L\n' r'refraction $\theta_\mathrm{L}$ (°)',
                  fontsize=fontsize_labels_pt)
    common_angle_plot_formatting(ax, theta_crit_deg)
    ax.set_yticks([0, 45, 90])
    ax.set_ylim(0, 95)


def D_SV_plot(ax, D_SV, theta_array_deg, theta_crit_deg,
              label_phase=False):
    real_and_imag_plots(ax, D_SV, theta_array_deg)
    abs_plot(ax, D_SV, theta_array_deg)
    ax.set_ylabel('Detection\ndirectivity ' r'$D_\mathrm{SV}$',
                  fontsize=fontsize_labels_pt)
    common_angle_plot_formatting(ax, theta_crit_deg)
    handle, ax_phase = phase_plot(ax, D_SV, theta_array_deg)
    if label_phase:
        ax_phase.set_ylabel(r'Phase angle ($\degree$)', color=color_phase,
                            fontsize=fontsize_labels_pt, x=1)
    ax.set_ylim(-2, 2)


def G_D_line_plot(ax, G_D, theta_array_deg, theta_crit_deg):
    real_and_imag_plots(ax, G_D, theta_array_deg)
    abs_plot(ax, G_D, theta_array_deg)
    ax.set_ylabel(r'$G_\mathrm{SV} D_\mathrm{SV}$',
                  fontsize=fontsize_labels_pt)
    common_angle_plot_formatting(ax, theta_crit_deg)
    ax.set_yticks(np.linspace(-3.5, 3.5, 9), minor=True)
    ax.set_yticks([-4, 0, 4])
    ax.set_ylim(-4, 4)
    phase_plot(ax, G_D, theta_array_deg)


def G_R_D_line_plot(ax, G_R_D, theta_array_deg, theta_crit_deg):
    real_and_imag_plots(ax, G_R_D, theta_array_deg)
    abs_plot(ax, G_R_D, theta_array_deg)
    ax.set_ylabel(r'$G_\mathrm{SV} R_\mathrm{SV-SV} D_\mathrm{SV}$',
                  fontsize=fontsize_labels_pt)
    common_angle_plot_formatting(ax, theta_crit_deg)
    ax.set_yticks(np.linspace(-3.5, 3.5, 9), minor=True)
    ax.set_yticks([-4, 0, 4])
    ax.set_ylim(-4, 4)
    phase_plot(ax, G_R_D, theta_array_deg)


def axis_label_line_plot(ax, label):
    ax.set_xlabel(label, fontsize=fontsize_labels_pt, labelpad=0)


def angle_line_label(ax, angle_deg, text):
    ax.text(angle_deg + 1, -1.9, text,
            fontsize=fontsize_labels_pt, va='bottom')


def angle_line_label_combined(ax, angle_deg, text):
    ax.text(angle_deg + 1, -3.8, text,
            fontsize=fontsize_labels_pt, va='bottom')


def ricker_modulation_plot(ax, ricker_mod_2d, axis_label_string,
                           c_map, theta_crit_deg, title_string,
                           ticklabels_tf=False,
                           ax_top=False):
    n_samples = np.shape(ricker_mod_2d)[0]
    im = ax.imshow(ricker_mod_2d, aspect='auto', cmap=c_map,
                   extent=(0, 90, n_samples, 0))
    im.set_clim(-0.5, 0.5)
    ax.axvline(theta_crit_deg, ls=my_dashes, lw=line_width_plots, color='w')
    ax.axvline(45, ls=my_dashes, lw=line_width_plots, color='w')
    ax.axvline(70, ls=my_dashes, lw=line_width_plots, color='w')
    ax.set_xlim(0, 90)
    ax.axhline(color='k', lw=0.8)
    ax.set_xticks(np.linspace(10, 80, 8), minor=True)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    # xticklabels must be drawn to ensure that axes with and without tick
    # labels have the same y-axis length.  If not desired, draw white
    # (invisible):
    if ticklabels_tf:
        ticklabel_color = 'k'
    else:
        ticklabel_color = 'w'

    if ax_top:
        ax.tick_params(which='both', top=True, labeltop=False,
                       bottom=False, labelbottom=False)
        ax.set_xlabel(axis_label_string,
                      fontsize=fontsize_labels_pt, labelpad=8)
    else:
        ax.tick_params(which='both', top=False, labeltop=False,
                       bottom=True, labelbottom=True,
                       labelcolor=ticklabel_color)
        ax.set_xlabel(axis_label_string,
                      fontsize=fontsize_labels_pt, labelpad=0)
    ax.set_xticks([0, 30, 60, 90], minor=False)
    ax.set_yticks([0])
    ax.set_ylabel('Time $t$', fontsize=fontsize_labels_pt)
    ax.set_title(title_string, fontsize=fontsize_labels_pt, loc='left', pad=10)
    return im


def set_mpl_fonts_Times_New_Roman():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.cal'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.default'] = 'it'
    plt.rcParams['mathtext.fontset'] = 'custom'


def plot_wiggles_DAS(ax, list_of_complex_sums, time_vector_us,
                     wiggle_labels, x_axis_amplitude, color_pos,
                     color_neg, plot_envelopes=False,
                     label_env_max=False, label_has=None,
                     n_decimals=2):
    # Use a global x_axis_amplitude to maintain scale across the wiggle
    # plots:
    ax.set_xlim(-x_axis_amplitude, x_axis_amplitude)
    # Generate x tick positions:
    n_wiggles = len(wiggle_labels)
    half_step = x_axis_amplitude/n_wiggles
    wiggle_centres = np.linspace(-x_axis_amplitude - half_step,
                                 x_axis_amplitude + half_step,
                                 n_wiggles + 2)[1:-1]
    ax.set_xticks(wiggle_centres, labels=wiggle_labels)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                   left=False, labelleft=False)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax.xaxis.set_label_position('top')
    ax.invert_yaxis()
    # Turn off left and right spines:
    ax.spines[['right', 'left']].set_visible(False)
    # Use list of label sides if provided:
    if not label_has:
        label_has = ['right'] * len(list_of_complex_sums)
    # Plot real part with color shading under curves:
    for sum_ds_complex, wiggle_centre, label_ha in zip(list_of_complex_sums,
                                                       wiggle_centres,
                                                       label_has):
        sum_real_shifted = np.real(sum_ds_complex) + wiggle_centre
        ax.plot(sum_real_shifted, time_vector_us,
                color='k', linewidth=0.7)
        ax.fill_betweenx(time_vector_us, wiggle_centre, sum_real_shifted,
                         where=(sum_real_shifted > wiggle_centre),
                         color=color_pos)
        ax.fill_betweenx(time_vector_us, wiggle_centre, sum_real_shifted,
                         where=(sum_real_shifted < wiggle_centre),
                         color=color_neg)
        # If requested, add evelope functions:
        if plot_envelopes:
            envelope_shifted = np.abs(sum_ds_complex) + wiggle_centre
            ax.plot(envelope_shifted, time_vector_us,
                    color=color_abs, lw=0.7, zorder=0)
            # If requested, label env maximum:
            if label_env_max:
                # Add line from peak of envelope function
                # down to a text annotation below:
                x_env_max = np.max(envelope_shifted)
                i_max = np.argmax(envelope_shifted)
                y_max = time_vector_us[i_max]
                y_text = (np.min(time_vector_us) +
                          ((np.max(time_vector_us) -
                            np.min(time_vector_us)) * 0.9))
                ax.vlines(x_env_max, y_max, y_text,
                          lw=line_width_plots, color='k')
                # Add text:
                env_max = np.max(np.abs(sum_ds_complex))
                string = f' {env_max:.{n_decimals}f} '
                ax.text(x_env_max, y_text, string, ha=label_ha,
                        fontsize=fontsize_labels_pt)


def plot_wiggles_DAS_central_20_percent(
    ax,
    list_of_complex_sums,
    time_vector_us,
    wiggle_labels,
    x_axis_amplitude,
    color_pos,
    color_neg,
    plot_envelopes=False,
    label_env_max=False,
    label_has=None,
    n_decimals=2,
    central_percent=20
):
    # Use a global x_axis_amplitude to maintain scale across the wiggle
    # plots:
    ax.set_xlim(-x_axis_amplitude, x_axis_amplitude)
    # Generate x tick positions:
    n_wiggles = len(wiggle_labels)
    half_step = x_axis_amplitude/n_wiggles
    wiggle_centres = np.linspace(-x_axis_amplitude - half_step,
                                 x_axis_amplitude + half_step,
                                 n_wiggles + 2)[1:-1]
    ax.set_xticks(wiggle_centres, labels=wiggle_labels)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                   left=False, labelleft=False)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax.xaxis.set_label_position('top')
    ax.invert_yaxis()
    # Turn off left and right spines:
    ax.spines[['right', 'left']].set_visible(False)
    # Use list of label sides if provided:
    if not label_has:
        label_has = ['right'] * len(list_of_complex_sums)
    # Plot real part with color shading under curves:
    for sum_ds_complex, wiggle_centre, label_ha in zip(list_of_complex_sums,
                                                       wiggle_centres,
                                                       label_has):
        sum_real_shifted = np.real(sum_ds_complex) + wiggle_centre
        ax.plot(sum_real_shifted, time_vector_us,
                color='k', linewidth=0.7)
        ax.fill_betweenx(time_vector_us, wiggle_centre, sum_real_shifted,
                         where=(sum_real_shifted > wiggle_centre),
                         color=color_pos)
        ax.fill_betweenx(time_vector_us, wiggle_centre, sum_real_shifted,
                         where=(sum_real_shifted < wiggle_centre),
                         color=color_neg)
        # If requested, add evelope functions:
        if plot_envelopes:
            envelope_shifted = np.abs(sum_ds_complex) + wiggle_centre
            ax.plot(envelope_shifted, time_vector_us,
                    color=color_abs, lw=0.7, zorder=0)
            # If requested, label env maximum within central 20%:
            if label_env_max:
                # Add line from max of envelope function in central 20%
                # down to a text annotation below:
                # Trim to central %:
                n_samples = len(envelope_shifted)
                proportion_centre = central_percent / 100
                proportion_centre_start = 0.5 - (proportion_centre / 2)
                proportion_centre_stop = 0.5 + (proportion_centre / 2)
                i_centre_start = int(proportion_centre_start * n_samples)
                i_centre_stop = int(proportion_centre_stop * n_samples)
                (envelope_shifted_central_20_percent
                 ) = envelope_shifted[i_centre_start:i_centre_stop]
                x_env_max_central = np.max(envelope_shifted_central_20_percent)
                i_max_central = i_centre_start + \
                    np.argmax(envelope_shifted_central_20_percent)
                y_max = time_vector_us[i_max_central]
                env_max_central = np.max(
                    np.abs(sum_ds_complex[i_centre_start:i_centre_stop]))
                string = f' {env_max_central:.{n_decimals}f} '

                if isinstance(label_ha, float):
                    factor_y_line = 0.8
                    x_text = x_env_max_central + label_ha
                    ha = 'center'
                else:
                    factor_y_line = 0.9
                    x_text = x_env_max_central
                    ha = label_ha
                y_line = (np.min(time_vector_us) +
                          ((np.max(time_vector_us) -
                            np.min(time_vector_us)) * factor_y_line))
                y_text = (np.min(time_vector_us) +
                          ((np.max(time_vector_us) -
                            np.min(time_vector_us)) * 0.9))
                ax.vlines(x_env_max_central, y_max, y_line,
                          lw=line_width_plots, color='k')
                ax.text(x_text, y_text, string, ha=ha,
                        fontsize=fontsize_labels_pt)


def plot_wiggles_VCF(ax, list_of_vcf_waveforms, time_vector_us,
                     wiggle_labels, x_axis_amplitude,
                     label_vcf_max=False, n_decimals=3):
    # Use a global x_axis_amplitude to maintain scale across the wiggle
    # plots:
    ax.set_xlim(-x_axis_amplitude, x_axis_amplitude)
    # Generate x tick positions:
    n_wiggles = len(wiggle_labels)
    half_step = x_axis_amplitude/n_wiggles
    wiggle_centres = np.linspace(-x_axis_amplitude - half_step,
                                 x_axis_amplitude + half_step,
                                 n_wiggles + 2)[1:-1]
    ax.set_xticks(wiggle_centres, labels=wiggle_labels)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                   left=False, labelleft=False)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax.xaxis.set_label_position('top')
    ax.invert_yaxis()
    # Turn off left and right spines:
    ax.spines[['right', 'left']].set_visible(False)
    # Compute y text:
    y_text = (np.min(time_vector_us) +
              ((np.max(time_vector_us) -
                np.min(time_vector_us)) * 0.5))
    # Plot VCF waveforms with color shading under the curves:
    for vcf_waveform, wiggle_centre in zip(list_of_vcf_waveforms,
                                           wiggle_centres):
        vcf_waveform_shifted = vcf_waveform + wiggle_centre
        # Black line plot:
        ax.plot(vcf_waveform_shifted, time_vector_us,
                color='k', linewidth=0.7)
        # Filled color under curve
        ax.fill_betweenx(time_vector_us, wiggle_centre, vcf_waveform_shifted,
                         where=(vcf_waveform_shifted > wiggle_centre),
                         color=color_vcf)
        # Add a vertical line marking the location of VCF=1 for this wiggle:
        x_vcf_1 = 1 + wiggle_centre
        ax.axvline(x_vcf_1, ls=my_dashes, color='k', lw=line_width_plots)
        # Add a vertical line at vcf=0:
        ax.axvline(wiggle_centre, color='k', lw=line_width_plots)
        # If requested, label maximum:
        if label_vcf_max:
            # Add text:
            vcf_max = np.max(vcf_waveform)
            x_text = vcf_max + wiggle_centre
            string = f'{vcf_max:.{n_decimals}f}  '
            ax.text(x_text, y_text, string, ha='right',
                    fontsize=fontsize_labels_pt, va='center')


def plot_VCF_argands(ax, list_of_row_samples_complex,
                     angular_range_labels, label_top_or_bottoms):
    # Compute the centre x coordinates of the argands that will
    # make them evenly spaced:
    n_argands = len(angular_range_labels)
    # Each argand has a width of 2 (-1 to +1), and the gap between argands
    # and at the ends will be 0.25.
    dx_gap = 0.5
    # Compute list of argand centre x coords:
    x_final_argand_centre = (dx_gap + 1) + ((n_argands - 1) * (dx_gap + 2))
    argand_centres = np.linspace((dx_gap + 1), x_final_argand_centre,
                                 n_argands)
    ax.set_xticks(argand_centres, labels=angular_range_labels)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                   left=False, labelleft=False)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax.xaxis.set_label_position('top')
    # Turn all spines:
    ax.spines[['right', 'left']].set_visible(False)
    # Iterate over row samples:
    for row_complex, x_argand_centre, label_tb in zip(
        list_of_row_samples_complex,
        argand_centres,
        label_top_or_bottoms
    ):
        # Add a vertical line marking the imaginary axis:
        grey = [0.7, 0.7, 0.7]
        ax.vlines(x_argand_centre, -1.2, 1.2, color=grey,
                  lw=line_width_plots)
        # Add a horizontal line representing the real axis:
        ax.hlines(0, x_argand_centre - 1.2, x_argand_centre + 1.2,
                  color=grey, lw=line_width_plots)
        # Normalise the magnitudes of the complex numbers in the row sample:
        row_complex_norm = row_complex / (np.abs(row_complex) *
                                          len(row_complex))
        row_cumsum = np.hstack(([0 + 0j], np.cumsum(row_complex_norm)))
        # Plot this cumsum, shifted horizontally by x_argand_centre:
        ax.plot(np.real(row_cumsum) + x_argand_centre,
                np.imag(row_cumsum),
                color='k', linewidth=0.7)
        # Add a unit circle centred on x_argand_centre:
        circ = patches.Circle((x_argand_centre, 0), 1,
                              lw=line_width_plots, ec='k',
                              fill=False)
        ax.add_patch(circ)
        # Add a circle of radius equal to the final VCF:
        vcf_final = np.abs(row_cumsum[-1])
        circ_vcf = patches.Circle((x_argand_centre, 0),
                                  vcf_final, lw=line_width_plots,
                                  ec=color_vcf, fill=False)
        ax.add_patch(circ_vcf)
        # Write the final_vcf value as text in the lower part of the Argand:
        ax.text(x_argand_centre, label_tb * 0.5, f'{vcf_final:.3f}',
                fontsize=fontsize_labels_pt, color='k', ha='center')
    # Set xlim:
    # Set equal data aspect ratio:
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')


def plot_travel_time_isochrones(
    travel_times_H_s,
    travel_times_S_direct_s,
    travel_times_hybrid_s
):
    # Visualise as an Imshow:
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, layout='constrained', figsize=(9, 3))
    gen_index = 80

    # Plot head wave travel times
    t_1_HW_us = travel_times_H_s[gen_index] / 10**-6
    ax1.imshow(t_1_HW_us)
    ax1.contour(t_1_HW_us, colors='k')
    ax1.set_title('Head wave')

    # Plot direct bulk SV wave travel times:
    t_2_SV_direct_us = travel_times_S_direct_s[gen_index] / 10**-6
    ax2.imshow(t_2_SV_direct_us)
    ax2.contour(t_2_SV_direct_us, colors='k')
    ax2.set_title('Direct bulk SV wave')

    # Plot hybrid travel times:
    t_3_hybrid_us = travel_times_hybrid_s[gen_index] / 10**-6
    ax3.imshow(t_3_hybrid_us)
    ax3.contour(t_3_hybrid_us, colors='k')
    ax3.set_title('Hybrid travel times')

    plt.show()


def plot_tfm_image(ax, intensity, dB_tf):
    pass


def plot_pcm(ax, pcm, v_max, ylabel=None, spines_color=None):
    im_pcm = ax.imshow(pcm, cmap=cc.m_CET_D7, vmax=v_max, vmin=-v_max)
    ax.set_xlabel('Generation index $g$', fontsize=fontsize_labels_pt)
    if ylabel:
        ax.set_ylabel('Detection index $d$', fontsize=fontsize_labels_pt)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax.xaxis.set_label_position('top')
    if spines_color:
        # Set colors:
        ax.spines['bottom'].set_color(spines_color)
        ax.spines['top'].set_color(spines_color)
        ax.spines['right'].set_color(spines_color)
        ax.spines['left'].set_color(spines_color)
        # Set line widths:
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)

    return im_pcm
