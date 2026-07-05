import numpy as np


def calculate_critical_angle_radians(v_l_mpers, v_t_mpers):
    angle_critical_radians = np.arcsin(v_t_mpers / v_l_mpers)
    return angle_critical_radians


def calculate_critical_angle_degrees(v_l_mpers, v_t_mpers):
    angle_critical_degrees = np.rad2deg(calculate_critical_angle_radians(v_l_mpers, v_t_mpers))
    return angle_critical_degrees
