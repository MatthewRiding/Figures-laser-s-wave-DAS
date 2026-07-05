import numpy as np


class PixelGrid():
    def __init__(self, z_start_mm, z_stop_mm, x_start_mm, x_stop_mm,
                 n_pixels_z):
        # Pin arguements to instance variables:
        self.z_start_mm = z_start_mm
        self.z_stop_mm = z_stop_mm
        self.x_start_mm = x_start_mm
        self.x_stop_mm = x_stop_mm
        self.n_pixels_z = n_pixels_z

        # Calculate grid sizes:
        self.grid_size_z_mm = np.abs(z_stop_mm - z_start_mm)

        # Calculate pixel size:
        self.pixel_size_mm = self.grid_size_z_mm / n_pixels_z

        # Build z-axis vector:
        self.z_vector_mm = np.linspace(start=self.z_start_mm, stop=self.z_stop_mm,
                                       num=self.n_pixels_z, endpoint=True)

        # Build x-axis vector:
        # Use of numpy arange() means x_end_mm will not be reached.
        # The highest x value will be the largest integer multiple of the step
        # size away from x_start_mm that is less than x_end_mm.
        self.x_vector_mm = np.arange(self.x_start_mm, self.x_stop_mm,
                                     self.pixel_size_mm)

    def get_meshgrid_arrays_m(self):
        # Create 2D mesh grid in metres:
        x_grid_m, z_grid_m = np.meshgrid(
            self.x_vector_mm * 10**-3, self.z_vector_mm * 10**-3)

        return x_grid_m, z_grid_m


class PixelGridXSymmetric(PixelGrid):
    def __init__(self, z_start_mm, z_stop_mm, grid_size_x_mm, n_pixels_z):
        x_start_mm = -(grid_size_x_mm / 2)
        x_stop_mm = grid_size_x_mm / 2
        super().__init__(z_start_mm, z_stop_mm, x_start_mm, x_stop_mm,
                         n_pixels_z)

    def get_z_mm_from_z_index(self, index_z_numpy):
        z_mm = self.z_vector_mm[index_z_numpy]
        return z_mm

    def get_x_mm_from_x_index(self, index_x_numpy):
        x_mm = self.x_vector_mm[index_x_numpy]
        return x_mm

    def get_imshow_extent(self):
        extent_image = (self.x_start_mm, self.x_stop_mm,
                        self.z_stop_mm, self.z_start_mm)
        return extent_image
