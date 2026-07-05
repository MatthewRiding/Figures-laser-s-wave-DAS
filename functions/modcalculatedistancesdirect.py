import numpy as np


def calculate_distances_direct_all_pixels_all_elements_m(x_grid_m, z_grid_m, x_elements_m):
    # Pre-allocate the 3-dimensional output array of distance values.
    # One page per element in the array, each page the shape of the imaging grid:
    n_tx = len(x_elements_m)
    shape = (n_tx,) + np.shape(x_grid_m)
    distances_direct_all_pixels_all_elements_m = np.zeros(shape)

    for element_index, x_element_m in enumerate(x_elements_m):
        distances_direct_all_pixels_all_elements_m[
        element_index, :, :] = calculate_distances_direct_all_pixels_single_element_m(x_grid_m, z_grid_m, x_element_m)

    return distances_direct_all_pixels_all_elements_m


def calculate_distances_direct_all_pixels_single_element_m(x_grid_m, z_grid_m, x_element_m):
    # Assumes a flat sample surface (all elements have z coordinate = 0).
    # Pythagoras' theorem:
    distances_direct_m = np.sqrt((x_grid_m - x_element_m) ** 2 + z_grid_m ** 2)
    return distances_direct_m
