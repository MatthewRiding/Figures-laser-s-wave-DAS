import numpy as np
from scipy.signal import hilbert

from functions.modbuildtfmgrid import build_tfm_grid
from corevariables.modwavesets import dict_wave_sets
from functions.modbuildxelements import build_x_elements_m
from functions.modfilterfmc3dbutter import filter_fmc3d_butter
from corevariables.modmaskbehaviours import dict_mask_behaviours
from functions.modcreatemasks import create_tfm_masks


def compute_tfm_complex(worker_id, fmc_3d, tfm_params, time_vector_us, signal_progress):
    # A generic TFM function that can run for different wave sets:
    signal_progress.emit((worker_id, ' Initialising...'))

    # Calculate additional parameters from supplied data:
    n_samples = np.shape(fmc_3d)[0]
    n_tx = np.shape(fmc_3d)[1]
    period_sampling_us = time_vector_us[1] - time_vector_us[0]
    frequency_sampling_hertz = 1 / (period_sampling_us * 10**-6)

    # Retrieve mask & polarity flipping functions if requested:
    if tfm_params.gen_mask_tf:
        # Some kind of mask has been requested.
        # Get the relevant MaskBehaviour:
        mask_behaviour = dict_mask_behaviours[tfm_params.mask_behaviour_string]
        # Get the mask creation, mask application and polarity flipping functions:
        numpy_masking_function = mask_behaviour.numpy_masking_function
        apply_mask_during_summing = mask_behaviour.apply_mask_during_summing
        flip_polarity_where_masked = mask_behaviour.flip_polarity_where_masked
    else:
        # No masking or polarity flipping requested.
        numpy_masking_function = None
        apply_mask_during_summing = False
        flip_polarity_where_masked = False

    # Build pixel co-ordinate mesh grid:
    x_grid_m, z_grid_m = build_tfm_grid(tfm_params.grid_size_x_mm, tfm_params.grid_size_z_mm, tfm_params.n_pixels_z)

    # Build a vector of the element x coordinates using pitch and n_tx:
    x_elements_m = build_x_elements_m(n_tx, tfm_params.pitch_mm)

    # Calculate travel times for send and receive legs:
    # Retrieve the travel time calculation function associated with the chosen wave set:
    wave_set = dict_wave_sets[tfm_params.wave_set_string]
    calculate_travel_times = wave_set.travel_time_function
    # All travel time calculation functions should have the same signature:
    times_send_s, times_receive_s = calculate_travel_times(x_grid_m, z_grid_m, x_elements_m,
                                                           tfm_params.v_l_mpers, tfm_params.v_t_mpers)
    # The arrays 'times_send_s' and 'times_receive_s' will both be 3-dimensional, with one page for each element in
    # the array, and each page the size of the imaging grid.  'times_send_s' may be a numpy MaskedArray.

    # If requested, create gen angle masks:
    if numpy_masking_function:
        gen_angles_rad_masked = create_tfm_masks(x_grid_m, z_grid_m, x_elements_m, tfm_params.mask_angle_deg,
                                                 numpy_masking_function)
    else:
        gen_angles_rad_masked = None

    # Apply bandpass filter to A-scans if requested:
    if tfm_params.filter_tf:
        signal_progress.emit((worker_id, ' Filtering...'))

        fmc_3d_filtered = filter_fmc3d_butter(fmc_3d, frequency_sampling_hertz, tfm_params.butter_order,
                                              band_min_MHz=tfm_params.band_min_MHz,
                                              band_max_MHz=tfm_params.band_max_MHz)

        # Use the analytic filtered fmc_3d in the subsequent calculations:
        fmc_3d_processed = hilbert(fmc_3d_filtered, axis=0)
    else:
        # No filtering: Use the analytic fmc_3d in the subsequent calculations:
        fmc_3d_processed = hilbert(fmc_3d, axis=0)

        # This function returns the filtered fmc_3d back to the caller.
        # When not filtering, we want to return the 'fmc_3d_filtered' variable as 'None':
        fmc_3d_filtered = None

    # Main imaging algorithm loop:

    # Pre-allocate an accumulator array for the intensity image:
    intensity_image_complex = np.zeros(np.shape(x_grid_m))

    # Nested loops over detection (slow, outer) and generation (fast, inner) index:
    for det_index in range(n_tx):
        # Emit the 'progress' signal, transmitting a string showing the current detection index out of the total:
        signal_progress.emit((worker_id, f' TFM ({det_index + 1}/{n_tx})...'))

        # Inner loop: over generation index (fast):
        for gen_index in range(n_tx):
            # Calculate total travel time for send and receive via each pixel for this combination of det_index and
            # gen_index:
            delays_for_this_a_scan_s = times_send_s[gen_index] + times_receive_s[det_index]

            # Submit the array of total travel times as 1D-interpolation query points for the A-scan associated with
            # this combination of gen_index and det_index:
            a_scan_amplitudes_analytic = fmc_3d_processed[:, det_index, gen_index]
            sampled_amps_complex = np.interp(delays_for_this_a_scan_s,
                                             (time_vector_us * 10**-6), a_scan_amplitudes_analytic, left=0, right=0)

            if gen_angles_rad_masked is not None:
                mask = np.ma.getmask(gen_angles_rad_masked[gen_index])
                # If requested, apply generation ray angle mask:
                if apply_mask_during_summing:
                    sampled_amps_complex_processed = np.ma.masked_where(mask, sampled_amps_complex, copy=False)
                # If requested, flip polarities where masked:
                if flip_polarity_where_masked:
                    # Masked values are 'True' in the mask array.  Build a polarity flip factor array, with the same
                    # shape as the TFM image, containing -1 at masked pixels, and 1 at un-masked pixels.
                    polarity_flip_factor = np.where(mask, -1, 1)
                    sampled_amps_complex_processed = sampled_amps_complex * polarity_flip_factor
            else:
                # The user has requested no masking or polarity flipping:
                sampled_amps_complex_processed = sampled_amps_complex

            # Sum the sampled amplitudes onto the intensity image accumulator array:
            # Fill any masked elements with zeros:
            intensity_image_complex = intensity_image_complex + np.ma.filled(sampled_amps_complex_processed,
                                                                             fill_value=0)

    # Main loop over A-scans complete.  Complex intensity image created.

    # # Convert intensity to root-power decibels by normalising relative to a maximum intensity value.
    # # To avoid saturation by SAW crosstalk, ignore pixels in the upper third of the image.
    # row_index_below_which_to_max = round(tfm_params.n_pixels_z / 3)
    # reference_amplitude_0dB = np.max(np.abs(intensity_image_complex[row_index_below_which_to_max:, :]))
    #
    # # Convert to root-power dB using this reference amplitude:
    # image_decibels = 20 * np.log10(np.abs(intensity_image_complex) / reference_amplitude_0dB)

    # Return the image in dB units back to the script calling this function.
    return intensity_image_complex, fmc_3d_filtered
