import numpy as np


def calculate_pixel_ray_angle_matrix_deg(x_pixel_m, z_pixel_m, x_matrix_m):
    opposite_m = np.abs(x_pixel_m - x_matrix_m)
    adjacent_m = z_pixel_m
    angle_matrix_deg = np.rad2deg(np.arctan2(opposite_m, adjacent_m))
    return angle_matrix_deg
