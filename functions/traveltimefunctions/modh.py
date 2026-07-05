import numpy as np

from functions.modcalculatedirectrayanglesradians import calculate_direct_ray_angles_radians
from functions.modcalculatecriticalangle import calculate_critical_angle_radians


def calculate_travel_times_head_wave_to_pixels_masked(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers, v_LSAW_mpers=None):
    # Calculate head wave travel times for all pixels:
    times_head_wave_to_pixel = calculate_travel_times_head_wave_to_pixels(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers, v_LSAW_mpers)

    # Apply a mask to these 'send' travel times, masking those pixels where direct send ray angle would be less than
    # the critical angle:

    # Calculate send ray angles:
    angles_direct_rays_radians = calculate_direct_ray_angles_radians(
        x_grid_m, z_grid_m, x_elements_m)
    # Calculate critical angle:
    angle_critical_radians = calculate_critical_angle_radians(
        v_L_mpers, v_T_mpers)

    # Mask the 'send' travel times: Pixels with rays above the critical angle are masked:
    travel_times_h_subcrit_masked = np.ma.masked_where(angles_direct_rays_radians < angle_critical_radians,
                                                       times_head_wave_to_pixel)

    return travel_times_h_subcrit_masked


def calculate_travel_times_head_wave_to_pixels(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers, v_LSAW_mpers=None):
    """To calculate the time taken for the head wave to reach this pixel, extrapolate from the pixel back up to
    the surface in a straight line at the critical angle to the surface normal.  The point where this angled
    line reaches the surface will be called the T-wave birth point.  The lateral distance between the T-wave
    birth point and the surface point directly above the pixel (x_pixel) will be called 'g_m'.  The delay law
    is computed by adding the time taken for a leaky surface wave (LSW) to travel from the source point to the
    T-wave birth point, and then for the T-wave to travel from the birth point to the pixel at the critical angle."""

    # If no distinct LSAW speed is given, take it to be equal to the bulk longitudinal wave speed:
    if not v_LSAW_mpers:
        v_LSAW_mpers = v_L_mpers

    v_T_over_v_L = v_T_mpers / v_L_mpers
    g_m = (v_T_over_v_L / np.sqrt(1 - v_T_over_v_L ** 2)) * z_grid_m

    # Pre-allocate travel times array, with one page per element:
    n_tx = len(x_elements_m)
    shape = (n_tx,) + np.shape(x_grid_m)
    times_head_wave_to_pixel = np.zeros(shape)

    for element_index, x_element_m in enumerate(x_elements_m):
        """Calculate lateral distance 'g'- the lateral distance between the pixel and the T-wave birth point.
        g = tan(theta_crit) * z_pixel_mm
        tan(theta_crit) = tan(arcsin(v_T/v_L)) = (v_T/v_L) / sqrt(1 - (v_T/v_L)**2)"""

        times_head_wave_to_pixel[element_index, :, :] = (((np.abs(x_grid_m - x_element_m) - g_m) / v_LSAW_mpers)
                                                         + (np.sqrt(g_m**2 + z_grid_m**2) / v_T_mpers))

    return times_head_wave_to_pixel
