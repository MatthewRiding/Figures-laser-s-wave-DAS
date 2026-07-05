import numpy as np


from functions.modcalculateanglematrix import calculate_pixel_ray_angle_matrix_deg


def generate_delay_matrix(x_pixel_m, z_pixel_m, listed_tfm_image):
    # Get the send and receive leg travel time calculation functions:
    wave_type_send = listed_tfm_image.tfm_constructor.wave_type_send
    wave_type_receive = listed_tfm_image.tfm_constructor.wave_type_receive
    calculate_delay_times_single_pixel_all_el_send_s = wave_type_send.func_calculate_delay_times_single_pixel_all_el_s
    calculate_delay_times_single_pixel_all_el_receive_s = wave_type_receive.func_calculate_delay_times_single_pixel_all_el_s

    # Calculate the send delays for all elements:
    send_delays_all_el_s = calculate_delay_times_single_pixel_all_el_send_s(x_pixel_m, z_pixel_m,
                                                                            listed_tfm_image.tfm_constructor)
    # Calculate the receive delays for all elements:
    receive_delays_all_el_s = calculate_delay_times_single_pixel_all_el_receive_s(x_pixel_m, z_pixel_m,
                                                                                  listed_tfm_image.tfm_constructor)

    # Use np.meshgrid to repeat the send and receive delay vectors into arrays:
    send_delays_matrix_s, receive_delays_matrix_s = np.meshgrid(send_delays_all_el_s, receive_delays_all_el_s)

    # Sum the send and receive delay matrices to obtain an n_elements by n_elements matrix of delays:
    delay_matrix_s = send_delays_matrix_s + receive_delays_matrix_s

    # Mask delay law based on gen and det mask_specs:
    if listed_tfm_image.tfm_constructor.mask_spec_gen or listed_tfm_image.tfm_constructor.mask_spec_det:
        # Get gen mask:
        if listed_tfm_image.tfm_constructor.mask_spec_gen:
            mask_spec_gen = listed_tfm_image.tfm_constructor.mask_spec_gen
            # Apply mask to the delay matrix based on send ray angle:
            gen_angle_matrix_deg = calculate_pixel_ray_angle_matrix_deg(x_pixel_m, z_pixel_m,
                                                                        listed_tfm_image.x_gen_matrix_m)
            # Use the logic of the chosen mask behaviour:
            numpy_masking_function_gen = mask_spec_gen.mask_behaviour.numpy_masking_function
            gen_angle_matrix_masked = numpy_masking_function_gen(gen_angle_matrix_deg,
                                                                 mask_spec_gen.mask_angle_deg)
            mask_matrix_gen = np.ma.getmask(gen_angle_matrix_masked)
        else:
            mask_matrix_gen = np.ma.nomask

        # Get det mask:
        if listed_tfm_image.tfm_constructor.mask_spec_det:
            mask_spec_det = listed_tfm_image.tfm_constructor.mask_spec_det
            # Apply mask to the delay matrix based on send ray angle:
            det_angle_matrix_deg = calculate_pixel_ray_angle_matrix_deg(x_pixel_m, z_pixel_m,
                                                                        listed_tfm_image.x_det_matrix_m)
            # Use the logic of the chosen mask behaviour:
            numpy_masking_function_det = mask_spec_det.mask_behaviour.numpy_masking_function
            det_angle_matrix_masked = numpy_masking_function_det(det_angle_matrix_deg,
                                                                 mask_spec_det.mask_angle_deg)
            mask_matrix_det = np.ma.getmask(det_angle_matrix_masked)
        else:
            mask_matrix_det = np.ma.nomask

        # Combine gen and det masks:
        mask_gen_and_det = np.ma.mask_or(mask_matrix_gen, mask_matrix_det)
        # Apply combined mask to delay matrix:
        delay_matrix_s = np.ma.masked_where(mask_gen_and_det, delay_matrix_s, copy=False)

    return delay_matrix_s