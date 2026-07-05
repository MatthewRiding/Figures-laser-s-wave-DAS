import numpy as np

from functions.modcalculatedistancesdirect import calculate_distances_direct_all_pixels_single_element_m
from functions.modcalculatedirectrayangles import calculate_direct_ray_angles_all_pixels_single_el_radians


def calculate_travel_times_all_pixels_single_el_direct_s(x_grid_m, z_grid_m, x_element_m, c_mpers):
    """Calculate direct straight-ray travel times to/from all pixels in the grid specified by tfm_constructor
    from/to the element at x_element_m, at the bulk elastic wave speed c_mpers."""
    # Calculate straight ray distances to all pixels from the element at x_element_m:
    distances_direct_all_pixels_m = calculate_distances_direct_all_pixels_single_element_m(x_grid_m, z_grid_m, x_element_m)
    # Divide straight-ray-distances by given bulk wave speed:
    travel_times_all_pixels_s = distances_direct_all_pixels_m / c_mpers
    return travel_times_all_pixels_s


def calculate_travel_times_all_pixels_single_el_l_direct_s(x_grid_m, z_grid_m, x_element_m, tfm_constructor):
    """Calculate direct bulk longitudinal wave travel times to/from all pixels in the specified grid,
    and from/to the element at x_element_m."""
    travel_times_l_all_pixels_s = calculate_travel_times_all_pixels_single_el_direct_s(x_grid_m, z_grid_m, x_element_m,
                                                                                       tfm_constructor.material.c_l_mpers)
    return travel_times_l_all_pixels_s


def calculate_travel_times_all_pixels_single_el_t_direct_s(x_grid_m, z_grid_m, x_element_m, tfm_constructor):
    """Calculate direct bulk transverse (shear) wave travel times to/from all pixels in the specified grid,
    and from/to the element at x_element_m."""
    travel_times_t_all_pixels_s = calculate_travel_times_all_pixels_single_el_direct_s(x_grid_m, z_grid_m, x_element_m,
                                                                                       tfm_constructor.material.c_t_mpers)
    return travel_times_t_all_pixels_s


def calculate_travel_times_all_pixels_single_el_head_wave_flat_surface_s(x_grid_m, z_grid_m, x_element_m, tfm_constructor):
    """To calculate the time taken for the head wave to reach this pixel:
     - Extrapolate from the pixel back up to the flat surface in a straight line at the critical angle to the surface normal.
     - The point where this angled line reaches the surface will be called the T-wave birth point.
     - The lateral distance between the T-wave birth point and the surface point directly above the pixel (x_pixel) will be called 'g_m'.
     - The total travel time is computed by adding the time taken for a leaky surface wave (LSW) to travel from the source point to the
       T-wave birth point, and then for the bulk T-wave to travel from the birth point to the pixel at the critical angle.
    This model assumes that the surface of the sample is perfectly flat.  Non-physical travel times are returned for
    pixels below the critical angle for a given generation point."""

    # Get data from tfm_constructor:
    c_t_over_c_l = tfm_constructor.material.c_t_mpers / tfm_constructor.material.c_l_mpers
    c_lsaw_mpers = tfm_constructor.material.c_lsaw_mpers
    c_t_mpers = tfm_constructor.material.c_t_mpers

    # Calculate lateral distance 'g'- the lateral distance between the pixel and the T-wave birth point.
    # g = tan(theta_crit) * z_pixel_mm
    # tan(theta_crit) = tan(arcsin(v_T/v_L)) = (v_T/v_L) / sqrt(1 - (v_T/v_L)**2)

    # tan(theta_crit) can be expressed more compactly using trig identities (WolframAlpha):

    g_m = (c_t_over_c_l / np.sqrt(1 - c_t_over_c_l ** 2)) * z_grid_m

    times_head_wave_to_pixel_s = (((np.abs(x_grid_m - x_element_m) - g_m) / c_lsaw_mpers) +
                                  (np.sqrt(g_m ** 2 + z_grid_m ** 2) / c_t_mpers))

    return times_head_wave_to_pixel_s


def calculate_travel_times_all_pixels_single_el_head_wave_subcrit_masked_s(x_grid_m, z_grid_m, x_element_m, tfm_constructor):
    """Calls 'calculate_travel_times_head_wave_to_pixels', and then applies a numpy mask to the pixels whose direct rays
    are below the critical angle."""
    # Calculate head wave travel times for all pixels:
    times_head_wave_to_pixel_s = calculate_travel_times_all_pixels_single_el_head_wave_flat_surface_s(x_grid_m, z_grid_m, x_element_m,
                                                                                                      tfm_constructor)

    # Apply a mask to these 'send' travel times, masking those pixels where direct send ray angle would be less than
    # the critical angle:

    # Calculate send ray angles:
    angles_direct_rays_radians = calculate_direct_ray_angles_all_pixels_single_el_radians(x_grid_m, z_grid_m, x_element_m)
    # Retrieve critical angle:
    angle_critical_radians = tfm_constructor.material.critical_angle_radians

    # Mask the 'send' travel times: Pixels with rays above the critical angle are masked:
    travel_times_h_subcrit_masked_s = np.ma.masked_where(angles_direct_rays_radians < angle_critical_radians,
                                                         times_head_wave_to_pixel_s)
    return travel_times_h_subcrit_masked_s


def calculate_travel_times_all_pixels_single_el_shear_hybrid_s(x_grid_m, z_grid_m, x_element_m, tfm_constructor):
    """This 'hybrid' travel time calculation uses direct shear travel times for pixels below the critical angle, and
    flat-surface shear head wave travel times for pixels above the critical angle from a given generation point."""

    # Compute the Head wave send travel time for all pixels (sub and supercrit):
    travel_times_h_s = calculate_travel_times_all_pixels_single_el_head_wave_flat_surface_s(x_grid_m, z_grid_m, x_element_m, tfm_constructor)

    # Compute direct ray SV travel time for all pixels:
    travel_times_sv_direct_s = calculate_travel_times_all_pixels_single_el_t_direct_s(x_grid_m, z_grid_m, x_element_m, tfm_constructor)

    # Compute direct ray angles and critical angle:
    angles_direct_rays_rad = calculate_direct_ray_angles_all_pixels_single_el_radians(x_grid_m, z_grid_m, x_element_m)
    theta_crit_rad = tfm_constructor.material.critical_angle_radians

    # Now combine them according to the ray angle threshold:  Pixels below theta_crit get the direct SV travel
    # time, whilst pixels above theta crit get the head wave travel time:
    travel_times_hybrid_send_s = np.where(angles_direct_rays_rad < theta_crit_rad, travel_times_sv_direct_s,
                                          travel_times_h_s)
    return travel_times_hybrid_send_s
