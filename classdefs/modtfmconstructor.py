from functions.modbuildtfmgrid import build_tfm_grid
from functions.modbuildxelements import build_x_elements_m
from functions.modcalculatedistancesdirect import calculate_distances_direct_all_pixels_all_elements_m
from functions.modcalculatedirectrayangles import (calculate_direct_ray_angles_all_pixels_all_el_radians,
                                                   calculate_direct_ray_angles_all_pixels_all_el_degrees)
from functions.modcreatemasks import create_tfm_masks


class TFMConstructor:
    def __init__(self,
                 image_name_string,
                 n_elements,
                 pitch_mm,
                 grid_size_x_mm,
                 grid_size_z_mm,
                 n_pixels_z,
                 material,
                 wave_type_send,
                 wave_type_receive,
                 filter_spec,
                 mask_spec_gen,
                 mask_spec_det
                 ):

        # Pin given values for variables to class instance as instance variables:
        self.n_elements = n_elements
        self.pitch_mm = pitch_mm
        self.grid_size_x_mm = grid_size_x_mm
        self.grid_size_z_mm = grid_size_z_mm
        self.n_pixels_z = n_pixels_z
        self.material = material
        self.wave_type_send = wave_type_send
        self.wave_type_receive = wave_type_receive
        self.filter_spec = filter_spec
        self.mask_spec_gen = mask_spec_gen
        self.mask_spec_det = mask_spec_det

        # Calculate some derived instance variables:
        # If a custom image name string has been give, assign it, else create a
        # default name using the given parameters.
        if image_name_string:
            self.image_name_string = image_name_string
        else:
            # No custom name given.  Build a default name:
            wave_set_string = wave_type_send.string_name + '-' + wave_type_receive.string_name
            filter_string = f' {filter_spec.band_min_mhz:g}-{filter_spec.band_max_mhz:g}MHz' if filter_spec else ''
            masked_string = ' masked' if mask_spec_gen or mask_spec_det else ''
            # Combine:
            self.image_name_string = wave_set_string + filter_string + masked_string

        # Build vector of element x-coordinates:
        self.x_elements_m = build_x_elements_m(self.n_elements, self.pitch_mm)

    def get_pixel_meshgrid_m(self):
        # Build pixel co-ordinate mesh grid:
        x_grid_m, z_grid_m = build_tfm_grid(self.grid_size_x_mm, self.grid_size_z_mm, self.n_pixels_z)
        return x_grid_m, z_grid_m

    def get_direct_ray_distances_all_pixels_all_el_m(self, x_grid_m, z_grid_m):
        distances_direct_all_pixels_all_el_m = calculate_distances_direct_all_pixels_all_elements_m(x_grid_m,
                                                                                                    z_grid_m,
                                                                                                    self.x_elements_m)
        return distances_direct_all_pixels_all_el_m

    def get_angles_direct_rays_all_pixels_all_el_radians(self, x_grid_m, z_grid_m):
        angles_direct_all_pixels_all_el_radians = calculate_direct_ray_angles_all_pixels_all_el_radians(x_grid_m, z_grid_m,
                                                                                                        self.x_elements_m)
        return angles_direct_all_pixels_all_el_radians

    def get_angles_direct_rays_all_pixels_all_el_degrees(self, x_grid_m, z_grid_m):
        angles_direct_all_pixels_all_el_degrees = calculate_direct_ray_angles_all_pixels_all_el_degrees(x_grid_m, z_grid_m,
                                                                                      self.x_elements_m)
        return angles_direct_all_pixels_all_el_degrees

    def get_travel_times_all_pixels_all_el(self, x_grid_m, z_grid_m):
        """
        Returns a tuple (times_send_s, times_receive_s) of two 3D ndarrays of shape (n_elements, n_pixels_z, n_pixels_x).
        For the send travel times array, each page contains the travel times (in seconds) to each pixel by the send wave
        type from the array element with the corresponding page index.
        For the receive travel times array, each page contains the travel times (in seconds) from each pixel by the
        receive wave type to the array element with the corresponding page index.
        """
        travel_times_all_pixels_all_el_send_s = self.wave_type_send.func_calculate_travel_times_all_pixels_all_el_s(
            x_grid_m, z_grid_m, self)
        travel_times_all_pixels_all_el_receive_s = self.wave_type_receive.func_calculate_travel_times_all_pixels_all_el_s(
            x_grid_m, z_grid_m, self)
        return travel_times_all_pixels_all_el_send_s, travel_times_all_pixels_all_el_receive_s

    def get_masks(self, x_grid_m, z_grid_m):
        # If either gen or det masks have been requested, create coordinate grids:
        if self.mask_spec_gen or self.mask_spec_det:
            # Gen masks:
            if self.mask_spec_gen:
                angles_all_pixels_all_el_rad_masked_gen = create_tfm_masks(x_grid_m, z_grid_m, self.x_elements_m,
                                                                           self.mask_spec_gen)
            else:
                angles_all_pixels_all_el_rad_masked_gen = None

            # Det masks:
            if self.mask_spec_det:
                angles_all_pixels_all_el_rad_masked_det = create_tfm_masks(x_grid_m, z_grid_m, self.x_elements_m,
                                                                           self.mask_spec_det)
            else:
                angles_all_pixels_all_el_rad_masked_det = None

            return angles_all_pixels_all_el_rad_masked_gen, angles_all_pixels_all_el_rad_masked_det
        else:
            return None, None

    def get_aperture_mm(self):
        aperture_mm = self.pitch_mm * (self.n_elements - 1)
        return aperture_mm


class TFMConstructorPixelGrid:
    def __init__(self,
                 image_name_string,
                 n_elements,
                 pitch_mm,
                 pixel_grid,
                 material,
                 wave_type_send,
                 wave_type_receive,
                 filter_spec,
                 mask_spec_gen,
                 mask_spec_det
                 ):

        # Pin given values for variables to class instance as instance variables:
        self.n_elements = n_elements
        self.pitch_mm = pitch_mm
        self.pixel_grid = pixel_grid
        self.material = material
        self.wave_type_send = wave_type_send
        self.wave_type_receive = wave_type_receive
        self.filter_spec = filter_spec
        self.mask_spec_gen = mask_spec_gen
        self.mask_spec_det = mask_spec_det

        # Calculate some derived instance variables:
        # If a custom image name string has been give, assign it, else create a
        # default name using the given parameters.
        if image_name_string:
            self.image_name_string = image_name_string
        else:
            # No custom name given.  Build a default name:
            wave_set_string = (wave_type_send.string_name +
                               '-' + wave_type_receive.string_name)
            filter_string = f' {filter_spec.band_min_mhz:g}-{filter_spec.band_max_mhz:g}MHz' if filter_spec else ''
            masked_string = ' masked' if mask_spec_gen or mask_spec_det else ''
            # Combine:
            self.image_name_string = (wave_set_string +
                                      filter_string +
                                      masked_string)

        # Build vector of element x-coordinates:
        self.x_elements_m = build_x_elements_m(self.n_elements, self.pitch_mm)

    def get_pixel_meshgrid_m(self):
        # Call relevant method of PixelGrid class:
        x_grid_m, z_grid_m = self.pixel_grid.get_meshgrid_arrays_m()
        return x_grid_m, z_grid_m

    def get_direct_ray_distances_all_pixels_all_el_m(self, x_grid_m, z_grid_m):
        distances_direct_all_pixels_all_el_m = calculate_distances_direct_all_pixels_all_elements_m(x_grid_m,
                                                                                                    z_grid_m,
                                                                                                    self.x_elements_m)
        return distances_direct_all_pixels_all_el_m

    def get_angles_direct_rays_all_pixels_all_el_radians(self, x_grid_m, z_grid_m):
        angles_direct_all_pixels_all_el_radians = calculate_direct_ray_angles_all_pixels_all_el_radians(x_grid_m, z_grid_m,
                                                                                                        self.x_elements_m)
        return angles_direct_all_pixels_all_el_radians

    def get_angles_direct_rays_all_pixels_all_el_degrees(self, x_grid_m, z_grid_m):
        angles_direct_all_pixels_all_el_degrees = calculate_direct_ray_angles_all_pixels_all_el_degrees(x_grid_m, z_grid_m,
                                                                                      self.x_elements_m)
        return angles_direct_all_pixels_all_el_degrees

    def get_travel_times_all_pixels_all_el(self, x_grid_m, z_grid_m):
        """
        Returns a tuple (times_send_s, times_receive_s) of two 3D ndarrays of shape (n_elements, n_pixels_z, n_pixels_x).
        For the send travel times array, each page contains the travel times (in seconds) to each pixel by the send wave
        type from the array element with the corresponding page index.
        For the receive travel times array, each page contains the travel times (in seconds) from each pixel by the
        receive wave type to the array element with the corresponding page index.
        """
        travel_times_all_pixels_all_el_send_s = self.wave_type_send.func_calculate_travel_times_all_pixels_all_el_s(
            x_grid_m, z_grid_m, self)
        travel_times_all_pixels_all_el_receive_s = self.wave_type_receive.func_calculate_travel_times_all_pixels_all_el_s(
            x_grid_m, z_grid_m, self)
        return travel_times_all_pixels_all_el_send_s, travel_times_all_pixels_all_el_receive_s

    def get_masks(self, x_grid_m, z_grid_m):
        # If either gen or det masks have been requested, create coordinate grids:
        if self.mask_spec_gen or self.mask_spec_det:
            # Gen masks:
            if self.mask_spec_gen:
                angles_all_pixels_all_el_rad_masked_gen = create_tfm_masks(x_grid_m, z_grid_m, self.x_elements_m,
                                                                           self.mask_spec_gen)
            else:
                angles_all_pixels_all_el_rad_masked_gen = None

            # Det masks:
            if self.mask_spec_det:
                angles_all_pixels_all_el_rad_masked_det = create_tfm_masks(x_grid_m, z_grid_m, self.x_elements_m,
                                                                           self.mask_spec_det)
            else:
                angles_all_pixels_all_el_rad_masked_det = None

            return angles_all_pixels_all_el_rad_masked_gen, angles_all_pixels_all_el_rad_masked_det
        else:
            return None, None

    def get_aperture_mm(self):
        aperture_mm = self.pitch_mm * (self.n_elements - 1)
        return aperture_mm
