import numpy as np


def calculate_angles_direct_rays_to_pixel_all_el_rad(x_pixel_m, z_pixel_m, x_elements_m):
    opposites_m = np.abs(x_pixel_m - x_elements_m)
    adjacent_m = z_pixel_m
    angles_rays_deg = np.arctan2(opposites_m, adjacent_m)
    return angles_rays_deg


def calculate_angles_direct_rays_to_pixel_all_el_deg(x_pixel_m, z_pixel_m, x_elements_m):
    angles_rays_deg = np.rad2deg(calculate_angles_direct_rays_to_pixel_all_el_rad(x_pixel_m, z_pixel_m, x_elements_m))
    return angles_rays_deg
