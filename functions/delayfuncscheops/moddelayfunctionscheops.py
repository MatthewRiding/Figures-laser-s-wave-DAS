import numpy as np

from functions.modcalculatedirectrayanglespixel import calculate_angles_direct_rays_to_pixel_all_el_rad


def calculate_travel_times_single_pixel_all_el_direct_s(x_pixel_m, z_pixel_m, x_elements_m, c_mpers):
    """Calculates one-way travel times from/to all elements to/from a pixel at bulk wave speed c_mpers."""
    distances_direct_m = np.sqrt((x_pixel_m - x_elements_m) ** 2 + z_pixel_m ** 2)
    travel_times_all_el_to_pixel_s = distances_direct_m / c_mpers
    return travel_times_all_el_to_pixel_s


def calculate_travel_times_single_pixel_all_el_l_s(x_pixel_m, z_pixel_m, tfm_constructor):
    """Calculates one-way travel times from/to all elements to/from a pixel at bulk longitudinal wave speed."""
    travel_times_all_el_to_pixel_l_s = calculate_travel_times_single_pixel_all_el_direct_s(x_pixel_m, z_pixel_m,
                                                                                           tfm_constructor.x_elements_m,
                                                                                           c_mpers=tfm_constructor.material.c_l_mpers)
    return travel_times_all_el_to_pixel_l_s


def calculate_travel_times_single_pixel_all_el_t_s(x_pixel_m, z_pixel_m, tfm_constructor):
    """Calculates one-way travel times from/to all elements to/from a pixel at bulk transverse wave speed."""
    travel_times_all_el_to_pixel_t_s = calculate_travel_times_single_pixel_all_el_direct_s(x_pixel_m, z_pixel_m,
                                                                                           tfm_constructor.x_elements_m,
                                                                                           c_mpers=tfm_constructor.material.c_t_mpers)
    return travel_times_all_el_to_pixel_t_s


def calculate_travel_times_single_pixel_all_el_h_s(x_pixel_m, z_pixel_m, tfm_constructor):
    """Calculates one-way travel times from all elements to a pixel by a transverse head wave, assuming a flat surface."""
    # Calculate delay for each combination of x_gen and x_det:
    c_t_over_c_l = tfm_constructor.material.c_t_mpers / tfm_constructor.material.c_l_mpers
    # g_m is the lateral distance between 
    g_m = (c_t_over_c_l / np.sqrt(1 - c_t_over_c_l ** 2)) * z_pixel_m
    travel_times_all_el_to_pixel_h_s = (
            ((np.abs(x_pixel_m - tfm_constructor.x_elements_m) - g_m) / tfm_constructor.material.c_lsaw_mpers)
            + (np.sqrt(g_m ** 2 + z_pixel_m ** 2) / tfm_constructor.material.c_t_mpers))
    return travel_times_all_el_to_pixel_h_s


def calculate_travel_times_single_pixel_all_el_shear_hybrid_s(x_pixel, z_pixel, tfm_constructor):
    """Calculates one-way travel times from all elements to a pixel by the shear-hybrid wavefront (direct below critical
    angle, transverse head wave above critical angle."""
    # Calculate direct transverse wave travel times for all elements:
    travel_times_direct_t_s = calculate_travel_times_single_pixel_all_el_t_s(x_pixel, z_pixel, tfm_constructor)
    # Calculate transverse head wave travel times for all elements:
    travel_times_h_s = calculate_travel_times_single_pixel_all_el_h_s(x_pixel, z_pixel, tfm_constructor)
    # Calculate direct ray angles to this pixel from all elements:
    angles_direct_rad = calculate_angles_direct_rays_to_pixel_all_el_rad(x_pixel, z_pixel, tfm_constructor.x_elements_m)
    # Build the final travel times array based on the critical angle threshold:
    travel_times_hybrid_send_s = np.where(angles_direct_rad < tfm_constructor.material.critical_angle_radians,
                                          travel_times_direct_t_s,
                                          travel_times_h_s)
    return travel_times_hybrid_send_s
