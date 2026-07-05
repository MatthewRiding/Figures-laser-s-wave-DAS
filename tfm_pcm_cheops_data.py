from functions.modbuildxgenandxdetmatrices import build_x_gen_and_x_det_matrices_m
from functions.delaylawfunctions.moddelaylawfunctions import calculate_delay_law_tt

# Input parameters:
# Array parameters:
pitch_mm = 0.2
n_elements_SDH = 101
# Time limits:
t_min_us = -0.5
t_max_us = 9.49
# Wave speeds:
c_l_mpers = 6370
c_t_mpers = 3130
c_lsaw_mpers = 6000

# Select 3 pixels for which the Cheops pyramid and PCM will be plotted:
xz_pixel_1_mm = (-5, 2)
xz_pixel_2_mm = (0, 4.3)
xz_pixel_3_mm = (7, 8)
# Calculate x_gen and x_det matrices for array:
x_gen_matrix_m, x_det_matrix_m = build_x_gen_and_x_det_matrices_m(
    n_elements_SDH, pitch_mm)

# Compute Delay matrices for chosen pixels:


def get_SS_delay_matrix(xz_pixel_mm):
    delay_matrix_s = calculate_delay_law_tt(xz_pixel_mm[0]*10**-3,
                                            xz_pixel_mm[1]*10**-3,
                                            x_gen_matrix_m, x_det_matrix_m,
                                            v_t_mpers=c_t_mpers)
    return delay_matrix_s


delay_matrix_1_s = get_SS_delay_matrix(xz_pixel_1_mm)
delay_matrix_2_s = get_SS_delay_matrix(xz_pixel_2_mm)
delay_matrix_3_s = get_SS_delay_matrix(xz_pixel_3_mm)
