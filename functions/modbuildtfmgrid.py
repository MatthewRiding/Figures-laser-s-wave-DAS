import numpy as np


def build_tfm_grid(grid_size_x_mm, grid_size_z_mm, n_pixels_z):
    # Build a TFM grid and return X and Z co-ordinate arrays in the shape of the image:

    # Calculate pixel size:
    pixel_size_mm = grid_size_z_mm / n_pixels_z

    # Build z-axis vector:
    z_vector_mm = np.linspace(start=0, stop=grid_size_z_mm, num=n_pixels_z, endpoint=True)

    # Build x-axis vector:
    x_start_mm = -(grid_size_x_mm / 2)
    x_end_mm = - x_start_mm
    # Use of numpy arange() means x_end_mm will not be reached.  The highest x value will be the largest integer
    # multiple of the step size away from x_start_mm that is less than x_end_mm.
    x_vector_mm = np.arange(x_start_mm, x_end_mm, pixel_size_mm)

    # Create 2D mesh grid in metres:
    x_grid_m, z_grid_m = np.meshgrid(x_vector_mm * 10**-3, z_vector_mm * 10**-3)

    return x_grid_m, z_grid_m
