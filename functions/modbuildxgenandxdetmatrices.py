import numpy as np


from functions.modbuildxelements import build_x_elements_m


def build_x_gen_and_x_det_matrices_m(n_tx, pitch_mm):
    x_elements_m = build_x_elements_m(n_tx, pitch_mm)
    x_gen_matrix_m, x_det_matrix_m = np.meshgrid(x_elements_m, x_elements_m)
    return x_gen_matrix_m, x_det_matrix_m
