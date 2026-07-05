import numpy as np
from scipy.signal import hilbert


def compute_tfm_complex(full_matrix, tfm_constructor, signal_progress=None, worker_id=None):
    """
    Computes an elastic wave reflectivity image via the delay-and-sum imaging algorithm known as the Total Focussing
    Method (TFM) (see Holmes, Caroline, Bruce W. Drinkwater, and Paul D. Wilcox. "Post-processing of the full matrix of
    ultrasonic transmit–receive array data for non-destructive evaluation." NDT & e International 38.8 (2005):
     701-711.).

    Parameters:
    -------
    :param full_matrix: FullMatrixLinearPeriodic
        An instance of the FullMatrixLinearPeriodic class containing the full matrix displacement, time and sampling
        information.
    :param tfm_constructor: TFMConstructor
        An instance of the TFMConstructor class containing all the data regarding the desired TFM image.
    :param signal_progress: Signal, optional
        A Qt Signal that can be emitted to communicate progress information back to the main GUI thread of a PySide6
        application.  An instance of any class that supports '.emit()' can be passed for use outside a GUI framework.
    :param worker_id: str, optional
        A unique uuid for this TFM calculation, used to identify progress signals sent back to the Chorus
        GUI thread during parallel processing.  Can be replaced

    Returns:
    -------
    :return: (summed_displacement_image_complex_nm: ndarray of complex128, displacements_3d_filtered: ndarray)
        The delay-and-sum image.  The complex form is returned for optional envelope processing down-stream.
        The 3d array of filtered displacements is also returned for reference.
    """

    if signal_progress:
        # Emit the progress signal with the string 'Initialising...' to tell the user that
        # the calculation has begun:
        signal_progress.emit((worker_id, ' Initialising...'))

    # Build 2D arrays storing the imaging grid pixel coordinates:
    x_grid_m, z_grid_m = tfm_constructor.get_pixel_meshgrid_m()

    # Calculate travel times for send and receive legs:
    times_send_s, times_receive_s = tfm_constructor.get_travel_times_all_pixels_all_el(x_grid_m, z_grid_m)

    # Get any masks if specs have been provided to the constructor:
    gen_angles_rad_masked, det_angles_rad_masked = tfm_constructor.get_masks(x_grid_m, z_grid_m)

    # Apply bandpass filter to A-scans if requested:
    if tfm_constructor.filter_spec:
        if signal_progress:
            signal_progress.emit((worker_id, ' Filtering...'))

        displacements_fmc_3d_filtered_nm = tfm_constructor.filter_spec.apply_to_fmc(full_matrix)

        # Use the analytic filtered fmc_3d in the subsequent calculations:
        displacements_fmc_3d_processed_nm = hilbert(displacements_fmc_3d_filtered_nm, axis=0)
    else:
        # No filtering: Use the analytic fmc_3d in the subsequent calculations:
        displacements_fmc_3d_processed_nm = hilbert(full_matrix.displacements_3d_nm, axis=0)

        # This function (compute_tfm) returns the filtered fmc_3d back to the caller for display in B-scan & fmc plots.
        # When not filtering, we want to return the 'displacements_fmc_3d_filtered_nm' variable as 'None':
        displacements_fmc_3d_filtered_nm = None

    # Main imaging algorithm loop:

    # Pre-allocate an accumulator array for the intensity image:
    summed_displacement_image_complex_nm = np.zeros(np.shape(x_grid_m))

    # Assumption: 'Square' full matrix: n_tx = n_rx = n_elements.
    n_tx = full_matrix.n_elements
    n_rx = n_tx

    # Nested loops over detection (slow, outer) and generation (fast, inner) index:
    for det_index in range(n_rx):
        if signal_progress:
            # Emit the 'progress' signal, transmitting a string showing the current detection index out of the total:
            signal_progress.emit((worker_id, f' TFM ({det_index + 1}/{n_rx})...'))

        # Inner loop: over generation index (fast):
        for gen_index in range(n_tx):
            # Calculate the total travel time for send and receive via each pixel for this combination of det_index and
            # gen_index:
            delays_for_this_a_scan_s = times_send_s[gen_index] + times_receive_s[det_index]

            # Submit the array of total travel times as interpolation query points for the A-scan associated with
            # this combination of gen_index and det_index:
            a_scan_displacements_analytic_nm = displacements_fmc_3d_processed_nm[:, det_index, gen_index]
            displacements_sampled_complex_nm = np.interp(delays_for_this_a_scan_s,
                                                         (full_matrix.time_vector_us * 10 ** -6),
                                                         a_scan_displacements_analytic_nm, left=0, right=0)

            # Apply gen and det angle masks, if any:
            if gen_angles_rad_masked is not None or det_angles_rad_masked is not None:
                # Some masking has been requested:

                # Get gen mask for this gen index:
                if gen_angles_rad_masked is not None:
                    mask_gen = np.ma.getmask(gen_angles_rad_masked[gen_index])
                else:
                    mask_gen = np.ma.nomask

                # Get det mask for this det index:
                if det_angles_rad_masked is not None:
                    mask_det = np.ma.getmask(det_angles_rad_masked[det_index])
                else:
                    mask_det = np.ma.nomask

                # Combine gen and det masks:
                mask_gen_and_det = np.ma.mask_or(mask_gen, mask_det)
                # Apply the combined mask to the pixel contributions:
                displacements_sampled_complex_processed_nm = np.ma.masked_where(mask_gen_and_det,
                                                                                displacements_sampled_complex_nm,
                                                                                copy=False)
            else:
                # The user has requested no masking:
                displacements_sampled_complex_processed_nm = displacements_sampled_complex_nm

            # Sum the sampled amplitudes onto the intensity image accumulator array:
            # Fill any masked elements with zeros:
            summed_displacement_image_complex_nm = summed_displacement_image_complex_nm + np.ma.filled(
                displacements_sampled_complex_processed_nm, fill_value=0)

    # Main loop over A-scans complete.  Complex summed displacement image created.

    # Return the summed displacement image in units of complex nanometres back to the script calling this function.
    return summed_displacement_image_complex_nm, displacements_fmc_3d_filtered_nm
