import numpy as np

from functions.modcalculatedirectrayangles import calculate_direct_ray_angles_all_pixels_all_el_radians


def create_tfm_masks(x_grid_m, z_grid_m, x_elements_m, mask_spec):
    # Calculate send ray angles:
    angles_direct_all_pixels_all_el_radians = calculate_direct_ray_angles_all_pixels_all_el_radians(x_grid_m, z_grid_m, x_elements_m)

    # Get the numpy masking function:
    numpy_masking_function = mask_spec.mask_behaviour.numpy_masking_function

    # Apply the mask:
    # First, see if the mask_behaviour is band-based (using a start and stop angle):
    if type(mask_spec.mask_angle_deg) is tuple:
        # A tuple of mask values has been provided: The mask spec is band-based:
        mask_angles_tuple = mask_spec.mask_angle_deg
        mask_angle_lower_rad = np.deg2rad(mask_angles_tuple[0])
        mask_angle_upper_rad = np.deg2rad(mask_angles_tuple[1])
        angles_all_pixels_all_el_rad_masked = numpy_masking_function(angles_direct_all_pixels_all_el_radians,
                                                                     mask_angle_lower_rad,
                                                                     mask_angle_upper_rad)
    else:
        # A single angle has been given: The mask is high or low-pass:
        mask_angle_deg = mask_spec.mask_angle_deg
        angles_all_pixels_all_el_rad_masked = numpy_masking_function(angles_direct_all_pixels_all_el_radians,
                                                                     np.deg2rad(mask_angle_deg))

    return angles_all_pixels_all_el_rad_masked
