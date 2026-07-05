import numpy as np


def calculate_direct_ray_angles_radians(x_grid_m, z_grid_m, x_elements_m):
    # Pre-allocate the 3-dimensional output array of distance values.
    # One page per element in the array, each page the shape of the imaging grid:
    n_tx = len(x_elements_m)
    shape = (n_tx,) + np.shape(x_grid_m)
    angles_rays_radians = np.zeros(shape)

    for element_index, x_element in enumerate(x_elements_m):
        opposite_m = np.abs(x_grid_m - x_element)
        adjacent_m = z_grid_m
        angles_rays_radians[element_index, :, :] = np.arctan2(opposite_m, adjacent_m)

    return angles_rays_radians
