import numpy as np


from functions.modcalculatedistancesdirect import calculate_distances_direct_m
from functions.traveltimefunctions.modldirect import calculate_travel_times_l_direct
from functions.traveltimefunctions.modtdirect import calculate_travel_times_t_direct
from functions.traveltimefunctions.modh import (calculate_travel_times_head_wave_to_pixels_masked,
                                                calculate_travel_times_head_wave_to_pixels)
from functions.modcalculatecriticalangle import calculate_critical_angle_radians
from functions.modcalculatedirectrayanglesradians import calculate_direct_ray_angles_radians
from functions.modmyplotfuncs import plot_travel_time_isochrones


def calculate_travel_times_ll(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # L-L requires direct send and direct return paths:
    distances_direct_m = calculate_distances_direct_m(x_grid_m, z_grid_m, x_elements_m)

    # Convert distance to time for send and receive:
    travel_times_l_direct = calculate_travel_times_l_direct(distances_direct_m, v_L_mpers)

    # Return (send, return) travel times:
    return travel_times_l_direct, travel_times_l_direct


def calculate_travel_times_tt(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # T-T requires direct send and direct return paths:
    distances_direct_m = calculate_distances_direct_m(x_grid_m, z_grid_m, x_elements_m)

    # Convert distance to time for send and receive:
    travel_times_t_direct = calculate_travel_times_t_direct(distances_direct_m, v_T_mpers)

    # Return (send, return) travel times:
    return travel_times_t_direct, travel_times_t_direct


def calculate_travel_times_lt(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # L-T requires direct send and direct return paths:
    distances_direct_m = calculate_distances_direct_m(x_grid_m, z_grid_m, x_elements_m)

    # Convert distance to time for send and receive:
    travel_times_l_direct = calculate_travel_times_l_direct(distances_direct_m, v_L_mpers)
    travel_times_t_direct = calculate_travel_times_t_direct(distances_direct_m, v_T_mpers)

    # Return (send, return) travel times:
    return travel_times_l_direct, travel_times_t_direct


def calculate_travel_times_tl(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # T-L requires direct send and direct return paths:
    distances_direct_m = calculate_distances_direct_m(x_grid_m, z_grid_m, x_elements_m)

    # Convert distance to time for send and receive:
    travel_times_t_direct = calculate_travel_times_t_direct(distances_direct_m, v_T_mpers)
    travel_times_l_direct = calculate_travel_times_l_direct(distances_direct_m, v_L_mpers)

    # Return (send, return) travel times:
    return travel_times_t_direct, travel_times_l_direct


def calculate_travel_times_ht(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # H-T requires head wave send and direct return paths:
    distances_direct_m = calculate_distances_direct_m(x_grid_m, z_grid_m, x_elements_m)

    # Times:
    travel_times_h_subcrit_masked = calculate_travel_times_head_wave_to_pixels_masked(x_grid_m, z_grid_m, x_elements_m,
                                                                               v_L_mpers, v_T_mpers)
    travel_times_t_direct = calculate_travel_times_t_direct(distances_direct_m, v_T_mpers)

    # Return (send, return) travel times:
    return travel_times_h_subcrit_masked, travel_times_t_direct


def calculate_travel_times_hl(x_grid_m, z_grid_m, x_elements_m, v_L_mpers, v_T_mpers):
    # H-L requires head wave send and direct return paths:
    distances_direct_m = calculate_distances_direct_m(x_grid_m, z_grid_m, x_elements_m)

    # Times:
    travel_times_h_subcrit_masked = calculate_travel_times_head_wave_to_pixels_masked(x_grid_m, z_grid_m, x_elements_m,
                                                                               v_L_mpers, v_T_mpers)
    travel_times_l_direct = calculate_travel_times_l_direct(distances_direct_m, v_L_mpers)

    # Return (send, return) travel times:
    return travel_times_h_subcrit_masked, travel_times_l_direct


def calculate_travel_times_hybrid_s(x_grid_m, z_grid_m, x_elements_m, c_L_mpers, c_S_mpers, c_LSAW_mpers=6000):  
    # Compute the Head wave send travel time for all pixels (sub and supercrit):
    travel_times_h_s = calculate_travel_times_head_wave_to_pixels(x_grid_m, z_grid_m, x_elements_m, c_L_mpers, c_S_mpers, c_LSAW_mpers)

    # Compute direct ray SV travel time for all pixels:
    distances_direct_m = calculate_distances_direct_m(x_grid_m, z_grid_m, x_elements_m)
    travel_times_sv_direct_s = calculate_travel_times_t_direct(distances_direct_m, c_S_mpers)

    # Now combine them according to the ray angle threshold:  Pixels below theta_crit get the direct SV travel
    # time, whilst pixels above theta crit get the head wave travel time:
    angles_direct_rays_deg = np.rad2deg(calculate_direct_ray_angles_radians(x_grid_m, z_grid_m, x_elements_m))
    theta_crit_deg = np.rad2deg(calculate_critical_angle_radians(c_L_mpers, c_S_mpers))
    travel_times_hybrid_send_s = np.where(angles_direct_rays_deg < theta_crit_deg, travel_times_sv_direct_s, travel_times_h_s)

    # Return (send, return) travel times:
    return travel_times_hybrid_send_s, travel_times_sv_direct_s
