import numpy as np


from functions.modcalculatedistancesdirect import calculate_distances_direct_all_pixels_all_elements_m
from functions.delayfuncstfm.modldirect import calculate_travel_times_l_direct
from functions.delayfuncstfm.modtdirect import calculate_travel_times_t_direct
from functions.delayfuncstfm.modh import (calculate_travel_times_head_wave_to_pixels_subcrit_masked,
                                          calculate_travel_times_head_wave_to_pixels)
from functions.modcalculatecriticalangle import calculate_critical_angle_radians
from functions.modcalculatedirectrayangles import calculate_direct_ray_angles_all_pixels_all_el_radians


def calculate_xz_travel_times_ll(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # L-L requires direct send and direct return paths:
    distances_direct_m = calculate_distances_direct_all_pixels_all_elements_m(x_grid_m, z_grid_m, x_elements_m)

    # Convert distance to time for send and receive:
    travel_times_l_direct = calculate_travel_times_l_direct(distances_direct_m, v_L_mpers)

    # Return (send, return) travel times:
    return travel_times_l_direct, travel_times_l_direct


def calculate_xz_travel_times_tt(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # T-T requires direct send and direct return paths:
    distances_direct_m = calculate_distances_direct_all_pixels_all_elements_m(x_grid_m, z_grid_m, x_elements_m)

    # Convert distance to time for send and receive:
    travel_times_t_direct = calculate_travel_times_t_direct(distances_direct_m, v_T_mpers)

    # Return (send, return) travel times:
    return travel_times_t_direct, travel_times_t_direct


def calculate_xz_travel_times_lt(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # L-T requires direct send and direct return paths:
    distances_direct_m = calculate_distances_direct_all_pixels_all_elements_m(x_grid_m, z_grid_m, x_elements_m)

    # Convert distance to time for send and receive:
    travel_times_l_direct = calculate_travel_times_l_direct(distances_direct_m, v_L_mpers)
    travel_times_t_direct = calculate_travel_times_t_direct(distances_direct_m, v_T_mpers)

    # Return (send, return) travel times:
    return travel_times_l_direct, travel_times_t_direct


def calculate_xz_travel_times_tl(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # T-L requires direct send and direct return paths:
    distances_direct_m = calculate_distances_direct_all_pixels_all_elements_m(x_grid_m, z_grid_m, x_elements_m)

    # Convert distance to time for send and receive:
    travel_times_t_direct = calculate_travel_times_t_direct(distances_direct_m, v_T_mpers)
    travel_times_l_direct = calculate_travel_times_l_direct(distances_direct_m, v_L_mpers)

    # Return (send, return) travel times:
    return travel_times_t_direct, travel_times_l_direct


def calculate_xz_travel_times_ht(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # H-T requires head wave send and direct return paths:
    distances_direct_m = calculate_distances_direct_all_pixels_all_elements_m(x_grid_m, z_grid_m, x_elements_m)

    # Times:
    travel_times_h_subcrit_masked = calculate_travel_times_head_wave_to_pixels_subcrit_masked(x_grid_m, z_grid_m, x_elements_m,
                                                                                              v_L_mpers, v_T_mpers)
    travel_times_t_direct = calculate_travel_times_t_direct(distances_direct_m, v_T_mpers)

    # Return (send, return) travel times:
    return travel_times_h_subcrit_masked, travel_times_t_direct


def calculate_xz_travel_times_hl(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # H-L requires head wave send and direct return paths:
    distances_direct_m = calculate_distances_direct_all_pixels_all_elements_m(x_grid_m, z_grid_m, x_elements_m)

    # Times:
    travel_times_h_subcrit_masked = calculate_travel_times_head_wave_to_pixels_subcrit_masked(x_grid_m, z_grid_m, x_elements_m,
                                                                                              v_L_mpers, v_T_mpers)
    travel_times_l_direct = calculate_travel_times_l_direct(distances_direct_m, v_L_mpers)

    # Return (send, return) travel times:
    return travel_times_h_subcrit_masked, travel_times_l_direct


def calculate_xz_travel_times_hybrid_s(x_grid_m, z_grid_m, x_elements_m, c_L_mpers, c_S_mpers, c_LSAW_mpers=None):
    """This delay function assumes a hybrid direct (below the critical angle) & head SV wave (above the critical angle)
    send travel time from the generation position to each pixel, which is added to a direct SV return travel time from
    each pixel to the detection position."""
    # Compute the Head wave send travel time for all pixels (sub- and super-crit):
    travel_times_h_s = calculate_travel_times_head_wave_to_pixels(x_grid_m, z_grid_m, x_elements_m, c_L_mpers,
                                                                  c_S_mpers, c_LSAW_mpers)

    # Compute direct ray SV travel time for all pixels:
    distances_direct_m = calculate_distances_direct_all_pixels_all_elements_m(x_grid_m, z_grid_m, x_elements_m)
    travel_times_sv_direct_s = calculate_travel_times_t_direct(distances_direct_m, c_S_mpers)

    # Now combine them according to the ray angle threshold:  Pixels below theta_crit get the direct SV travel
    # time, whilst pixels above theta crit get the head wave travel time:
    angles_direct_rays_deg = np.rad2deg(calculate_direct_ray_angles_all_pixels_all_el_radians(x_grid_m, z_grid_m, x_elements_m))
    theta_crit_deg = np.rad2deg(calculate_critical_angle_radians(c_L_mpers, c_S_mpers))
    travel_times_hybrid_send_s = np.where(angles_direct_rays_deg < theta_crit_deg, travel_times_sv_direct_s, travel_times_h_s)

    # Return (send, return) travel times:
    return travel_times_hybrid_send_s, travel_times_sv_direct_s
