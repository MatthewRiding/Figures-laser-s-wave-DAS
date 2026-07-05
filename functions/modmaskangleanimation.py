import matplotlib.pyplot as plt
import numpy as np
import colorcet as cc
from matplotlib import colormaps as mpl_cmaps
from matplotlib.patches import Circle

from corevariables.modieeeplotting import (IEEE_two_col_width_in,
                                           fontsize_labels_pt,
                                           my_dashes,
                                           line_width_plots)
from functions.modmyplotfuncs import set_mpl_fonts_Times_New_Roman
from functions.modextractpcm import extract_pcm
from functions.modbuildxgenandxdetmatrices import build_x_gen_and_x_det_matrices_m
from functions.modthetacrittrig import calculate_theta_crit_deg
from corevariables.modwavesets import dict_wave_sets


def plot_array_elements(ax, n_elements, pitch_mm):
    # We'll plot every other element (there are too many to plot them all)
    # as a magenta square at the top of the image:
    n_el_to_plot = int(np.round(n_elements / 2))
    aperture_mm = pitch_mm * (n_elements - 1)
    dx_elements_from_end_mm = np.linspace(0, aperture_mm, n_el_to_plot)
    # Centre array around x=0:
    x_elements_mm = (dx_elements_from_end_mm -
                     (np.max(dx_elements_from_end_mm) / 2))
    # Create a vector of constant z-coordinates:
    z_element_square_mm = -0.1
    z_vector_mm = np.ones(n_el_to_plot) * z_element_square_mm
    ax.plot(x_elements_mm, z_vector_mm, ls='none', marker='s',
            color='magenta', ms=1, clip_on=False)


def plot_zero_image(ax, c_map, x_width_mm, z_max_mm, n_pixels_z):
    x_max_mm = x_width_mm / 2
    extent = (-x_max_mm, x_max_mm, z_max_mm, 0)
    # Calculate n_pixels x (Roughly, the image will be overwritten):
    pixel_side_length_mm = z_max_mm / n_pixels_z
    n_pixels_x = int(np.round(x_width_mm / pixel_side_length_mm))
    zero_array = np.zeros((n_pixels_z, n_pixels_x))
    im = ax.imshow(zero_array, cmap=c_map, extent=extent)
    return im


def common_xz_ax_formatting(ax, n_elements, pitch_mm):
    plot_array_elements(ax, n_elements, pitch_mm)
    ax.set_xlabel(r'$x$ (mm)', fontsize=fontsize_labels_pt)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax.xaxis.set_label_position('top')


def set_ax1_title(ax, mask_angle_deg):
    ax.set_title(f'Mask angle = {mask_angle_deg:.0f}°',
                 fontsize=fontsize_labels_pt)


def set_up_ax1(ax, x_width_mm, z_max_mm, x_pixel_mm, z_pixel_mm,
               theta_crit_deg):
    # Set axes limits:
    x_max_mm = x_width_mm / 2
    ax.set_xlim(-x_max_mm, x_max_mm)
    ax.set_ylim(z_max_mm, 0)
    ax.set_aspect('equal')
    # Draw circle representing SDH:
    z_sdh_mm = 5
    x_sdh_mm = 9.75 - x_max_mm
    radius_sdh_mm = 0.5
    patch_circle = Circle((x_sdh_mm, z_sdh_mm), radius_sdh_mm,
                          ls='-', fill=False, ec='k', lw=0.8)
    ax.add_patch(patch_circle)
    # Set title:
    set_ax1_title(ax, 0)

    # Label y-axis:
    ax.set_ylabel(r'$z$ (mm)', fontsize=fontsize_labels_pt)

    # Plot a blue square at the chosen pixel location:
    ax.plot(x_pixel_mm, z_pixel_mm, ls='none',
            marker='s', mec='b', mfc='none', ms=2)

    # Create the mask angle line:
    x_coords_mm = [x_pixel_mm, x_pixel_mm, x_pixel_mm]
    z_coords_mm = [0, z_pixel_mm, 0]
    line_mask_angle, = ax.plot(x_coords_mm, z_coords_mm,
                               ls=my_dashes, color='k',
                               lw=line_width_plots)
    # Create mask angle lines at the critical angle:
    line_mask_angle_crit, = ax.plot([1, 1, 1], z_coords_mm, ls=':',
                                    color=(0.8, 0.8, 0.8),
                                    lw=line_width_plots)
    update_mask_lines_ax1(ax, line_mask_angle_crit, theta_crit_deg,
                          x_pixel_mm, z_pixel_mm)
    return line_mask_angle


def add_cbar(fig, ax, im, label_string):
    cax = ax.inset_axes([0.15, -0.25, 0.7, 0.2])
    cbar = fig.colorbar(im, ax=ax, location='bottom', cax=cax)
    cbar.set_label(label_string, fontsize=fontsize_labels_pt)
    cbar.ax.tick_params(labelsize=fontsize_labels_pt)


def add_pcm_axes(fig, pcm):
    # Create axis within figure:
    fig_width_in = fig.get_figwidth()
    fig_height_in = fig.get_figheight()
    ax_side_in = 1
    ax_width_norm = (ax_side_in / fig_width_in)
    ax_height_norm = (ax_side_in / fig_height_in)
    rect_norm = (0.14, 0.04, ax_width_norm, ax_height_norm)
    ax_pcm = fig.add_axes(rect_norm, in_layout=False)
    # Set axis parameters:
    ax_pcm.text(0.5, 1.1, 'Gen index', fontsize=fontsize_labels_pt,
                ha='center', transform=ax_pcm.transAxes)
    ax_pcm.text(-0.1, 0.5, 'Det index', fontsize=fontsize_labels_pt,
                va='center', ha='right', rotation='vertical',
                transform=ax_pcm.transAxes)
    ax_pcm.tick_params(top=True, labeltop=True, bottom=False,
                       labelbottom=False)
    ax_pcm.tick_params(axis='both', which='major', labelsize=8, pad=1)
    ax_pcm.xaxis.set_label_position('top')
    ticks = [0, 100]
    ax_pcm.set_yticks(ticks)
    ax_pcm.set_xticks(ticks)
    # Imshow the PCM on the pcm_axis:
    lim = 3
    ax_pcm.imshow(pcm*1000, cmap=cc.m_CET_D7, vmin=-lim, vmax=lim)
    # Plot a line on the PCM that will enclose the parts of the pcm
    # currently under the mask angle:
    line_mask_boundary, = ax_pcm.plot([0], [0], lw=line_width_plots, color='w',
                                      ls='-')
    fig.text(0.125, 0.37, 'PCM:', ha='right', fontsize=fontsize_labels_pt)
    return ax_pcm, line_mask_boundary


def set_up_row_2_axes(axs_second_row):
    for ax in axs_second_row:
        ax.tick_params(top=False, labeltop=False, bottom=False,
                       labelbottom=False)
        ax.set_axis_off()


def initialise_figure(n_elements, pitch_mm, x_width_mm, z_max_mm, n_pixels_z,
                      x_pixel_mm, z_pixel_mm, pcm, theta_crit_deg):
    # Set Matplotlib rcparams:
    set_mpl_fonts_Times_New_Roman()
    # Instantiate figure & axes:
    (fig,
     ((ax1, ax2, ax3),
      axs_secondrow)) = plt.subplots(2, 3,
                                     height_ratios=[1, 0.2],
                                     layout='constrained',
                                     figsize=(IEEE_two_col_width_in, 3.3))
    # Set up second row axes:
    set_up_row_2_axes(axs_secondrow)
    # Set up ax1:
    common_xz_ax_formatting(ax1, n_elements, pitch_mm)
    line_mask_angle = set_up_ax1(ax1, x_width_mm, z_max_mm,
                                 x_pixel_mm, z_pixel_mm,
                                 theta_crit_deg)

    # Set up ax2:
    common_xz_ax_formatting(ax2, n_elements, pitch_mm)
    im2 = plot_zero_image(ax2, cc.m_CET_D7,
                          x_width_mm, z_max_mm, n_pixels_z)
    im2.set_clim(vmin=-1, vmax=1)
    ax2.set_title('TFM image: summed displacements',
                  fontsize=fontsize_labels_pt)
    ax2.axes.yaxis.set_ticklabels([])
    add_cbar(fig, ax2, im2, 'Summed displacement (V)')

    # Set up ax3:
    common_xz_ax_formatting(ax3, n_elements, pitch_mm)
    im3 = plot_zero_image(ax3, mpl_cmaps['viridis'],
                          x_width_mm, z_max_mm, n_pixels_z)
    im3.set_clim(vmin=-15, vmax=0)
    ax3.set_title('TFM image: abs dB',
                  fontsize=fontsize_labels_pt)
    ax3.axes.yaxis.set_ticklabels([])
    add_cbar(fig, ax3, im3, 'Abs(analytic) (dB)')

    # Add a PCM axes below ax1:
    ax_pcm, line_mask_boundary = add_pcm_axes(fig, pcm)

    return fig, (ax1, ax2, ax3), (im2, im3), line_mask_angle, ax_pcm, line_mask_boundary


def update_mask_lines_ax1(ax, mask_line, mask_angle_deg,
                          x_pixel_mm, z_pixel_mm):
    # Calculate the lateral offset from the x_coordinate of the pixel to
    # the surface intercept of a line at the angle theta:
    if mask_angle_deg == 90:
        x_coords_mm = [-15, x_pixel_mm, 15]
        z_coords_mm = [z_pixel_mm, z_pixel_mm, z_pixel_mm]
        mask_line.set_data(x_coords_mm, z_coords_mm)
    else:
        offset_theta = np.tan(np.deg2rad(mask_angle_deg)) * z_pixel_mm
        x_coords_mm = [x_pixel_mm - offset_theta,
                       x_pixel_mm,
                       x_pixel_mm + offset_theta]
        mask_line.set_xdata(x_coords_mm)


def get_pcm(fmc_3d, time_vector_us, tfm_params, x_pixel_m, z_pixel_m):
    # Which wave set is associated with the current TFM image?
    wave_set_string = tfm_params.wave_set_string
    # Get the delay law function for this wave set:
    calculate_delay_law = dict_wave_sets[wave_set_string].delay_law_function
    # Build x_coordinate matrices:
    n_tx = np.shape(fmc_3d)[2]
    (x_gen_matrix_m,
     x_det_matrix_m) = build_x_gen_and_x_det_matrices_m(n_tx,
                                                        tfm_params.pitch_mm)
    # Calculate theta_crit:
    theta_crit_rad = np.deg2rad(calculate_theta_crit_deg(tfm_params.v_l_mpers,
                                                         tfm_params.v_t_mpers))
    # Call the delay law function and return an n_tx by n_tx matrix of delays:
    delay_matrix_s = calculate_delay_law(x_pixel_m, z_pixel_m,
                                         x_gen_matrix_m,
                                         x_det_matrix_m,
                                         theta_crit_rad,
                                         tfm_params.v_l_mpers,
                                         tfm_params.v_t_mpers)
    # Call extract PCM:
    pcm_complex = extract_pcm(fmc_3d, time_vector_us, delay_matrix_s)
    return pcm_complex


def update_line_mask_boundary(line_mask_boundary, mask_angle_deg,
                              z_pixel_mm, n_pitches_per_mm, el_pix):
    # Calculate the 'opposite' of the right angled triangle (x_pix, z_pix),
    # (x_pix, 0), and the point where a line drawn from the pixel to the
    # surface at the mask angle emerges:
    d_x_mask_mm = np.tan(np.deg2rad(mask_angle_deg)) * z_pixel_mm
    # Convert to 'element' coordinates:
    d_el_mask = d_x_mask_mm * n_pitches_per_mm
    # Calculate the lower and upper element limits of the mask for this pixel:
    el_lower = el_pix - d_el_mask
    el_upper = el_pix + d_el_mask
    # Create the x and y coordinates of the corners of the box:
    x_coords = [el_lower, el_upper, el_upper, el_lower, el_lower]
    y_coords = [el_lower, el_lower, el_upper, el_upper, el_lower]
    # Update the line data:
    line_mask_boundary.set_data(x_coords, y_coords)
