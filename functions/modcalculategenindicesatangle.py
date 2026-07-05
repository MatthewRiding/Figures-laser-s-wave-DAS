import numpy as np


def calculate_gen_indices_at_angle(x_pixel_m, z_pixel_m, angle_radians, pitch_mm, n_tx):
    # Use trigonometry to calculate the surface 'opposite' distance corresponding to the specified angle from the
    # selected pixel:
    opp_m = np.tan(angle_radians) * z_pixel_m
    x_lower_m = x_pixel_m - opp_m
    x_upper_m = x_pixel_m + opp_m
    # Convert from x position to linear periodic array element index using knowledge of the pitch:
    index_at_x_zero = (n_tx - 1) / 2  # Python is zero-indexed: array elements go from 0 to (n_tx-1).
    index_lower = index_at_x_zero + (x_lower_m / (pitch_mm * 10 ** -3))
    index_upper = index_at_x_zero + (x_upper_m / (pitch_mm * 10 ** -3))
    return index_lower, index_upper
