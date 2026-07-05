import numpy as np


def calculate_delay_law_direct(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m, v_send_mpers, v_return_mpers):
    # Calculate delay for each combination of x_gen and x_det:
    distance_send_m = np.sqrt((x_pixel_m - x_gen_matrix_m)**2 + z_pixel_m**2)
    distance_return_m = np.sqrt((x_pixel_m - x_det_matrix_m)**2 + z_pixel_m**2)
    delay_matrix_s = (distance_send_m / v_send_mpers) + (distance_return_m / v_return_mpers)
    return delay_matrix_s


def calculate_delay_law_ll(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                           angle_critical_radians, v_l_mpers, v_t_mpers):
    delay_matrix_s = calculate_delay_law_direct(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                                                v_send_mpers=v_l_mpers,
                                                v_return_mpers=v_l_mpers)
    return delay_matrix_s


def calculate_delay_law_tt(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                           angle_critical_radians=None, v_l_mpers=None, v_t_mpers=None):
    delay_matrix_s = calculate_delay_law_direct(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                                                v_send_mpers=v_t_mpers,
                                                v_return_mpers=v_t_mpers)
    return delay_matrix_s


# def calculate_delay_law_tt_subcrit(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
#                                    angle_critical_radians, v_l_mpers, v_t_mpers):
#     delay_matrix_s = calculate_delay_law_direct(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
#                                                 v_send_mpers=v_t_mpers,
#                                                 v_return_mpers=v_t_mpers)
#     # Apply mask based on send ray angle:
#     angles_send_rays_radians = calculate_angle_send_ray_radians(x_pixel_m, z_pixel_m, x_gen_matrix_m)
#     delay_matrix_masked_s = np.ma.masked_where(angles_send_rays_radians > angle_critical_radians, delay_matrix_s)
#     return delay_matrix_masked_s


def calculate_delay_law_lt(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                           angle_critical_radians, v_l_mpers, v_t_mpers):
    delay_matrix_s = calculate_delay_law_direct(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                                                v_send_mpers=v_l_mpers,
                                                v_return_mpers=v_t_mpers)
    return delay_matrix_s


def calculate_delay_law_tl(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                           angle_critical_radians, v_l_mpers, v_t_mpers):
    delay_matrix_s = calculate_delay_law_direct(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                                                v_send_mpers=v_t_mpers,
                                                v_return_mpers=v_l_mpers)
    return delay_matrix_s


def calculate_delay_law_h_then_direct(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m, angle_critical_radians,
                                      v_l_mpers, v_t_mpers, v_return_mpers):
    # Calculate delay for each combination of x_gen and x_det:
    v_t_over_v_l = v_t_mpers / v_l_mpers
    # g_m is the lateral distance between 
    g_m = (v_t_over_v_l / np.sqrt(1 - v_t_over_v_l ** 2)) * z_pixel_m
    time_h_send = (((np.abs(x_pixel_m - x_gen_matrix_m) - g_m) / v_l_mpers)
                   + (np.sqrt(g_m**2 + z_pixel_m**2) / v_t_mpers))
    distance_return_m = np.sqrt((x_pixel_m - x_det_matrix_m) ** 2 + (z_pixel_m ** 2))
    delay_matrix_s = time_h_send + (distance_return_m / v_return_mpers)
    # # Determine validity mask based on send ray angle:
    # angle_send_ray = calculate_angle_send_ray_radians(x_pixel_m, z_pixel_m, x_gen_matrix_m)
    # # Use send angle to apply a mask to the delay_matrix:
    # delay_matrix_masked_s = np.ma.masked_where(angle_send_ray < angle_critical_radians, delay_matrix_s)
    return delay_matrix_s


def calculate_delay_law_hl(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                           angle_critical_radians, v_l_mpers, v_t_mpers):
    delay_matrix_s = calculate_delay_law_h_then_direct(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                                                       angle_critical_radians, v_l_mpers, v_t_mpers,
                                                       v_return_mpers=v_l_mpers)
    return delay_matrix_s


def calculate_delay_law_ht(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                           angle_critical_radians, v_l_mpers, v_t_mpers):
    delay_matrix_s = calculate_delay_law_h_then_direct(x_pixel_m, z_pixel_m, x_gen_matrix_m, x_det_matrix_m,
                                                       angle_critical_radians, v_l_mpers, v_t_mpers,
                                                       v_return_mpers=v_t_mpers)
    return delay_matrix_s
