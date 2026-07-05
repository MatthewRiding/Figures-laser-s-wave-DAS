import numpy as np


def calculate_direct_ray_angles_all_pixels_all_el_radians(x_grid_m, z_grid_m, x_elements_m):
    # Pre-allocate the 3-dimensional output array of distance values.
    # One page per element in the array, each page the shape of the imaging grid:
    n_tx = len(x_elements_m)
    shape = (n_tx,) + np.shape(x_grid_m)
    angles_rays_all_pixels_all_el_radians = np.zeros(shape)

    for element_index, x_element_m in enumerate(x_elements_m):
        angles_rays_all_pixels_all_el_radians[
        element_index, :, :] = calculate_direct_ray_angles_all_pixels_single_el_radians(x_grid_m, z_grid_m, x_element_m)

    return angles_rays_all_pixels_all_el_radians


def calculate_direct_ray_angles_all_pixels_all_el_degrees(x_grid_m, z_grid_m, x_elements_m):
    angles_direct_rays_all_pixels_all_el_radians = calculate_direct_ray_angles_all_pixels_all_el_radians(x_grid_m, z_grid_m, x_elements_m)
    angles_direct_rays_all_pixels_all_el_degrees = np.rad2deg(angles_direct_rays_all_pixels_all_el_radians)
    return angles_direct_rays_all_pixels_all_el_degrees


def calculate_direct_ray_angles_all_pixels_single_el_radians(x_grid_m, z_grid_m, x_element_m):
    opposite_m = np.abs(x_grid_m - x_element_m)
    adjacent_m = z_grid_m
    angles_rays_all_pixels_radians = np.arctan2(opposite_m, adjacent_m)
    return angles_rays_all_pixels_radians
