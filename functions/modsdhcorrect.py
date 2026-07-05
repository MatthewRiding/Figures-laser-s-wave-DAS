import numpy as np
from functions.modthetacrittrig import tan_theta_crit
from corevariables.modsdhexperiment import (
    material_al_sdh,
    params_al_sdh)


def sdh_correct_t_0_based(b_scan_ds, t_0_echo_us, t_0_half_width_us, n_t_0s,
                          x_gen_det_vector_mm, time_vector_us, c_mpers,
                          x_sdh_mm, radius_sdh_mm):
    # Generate vector of t_0 values to evaluate delay laws for:
    t_0_min_us = t_0_echo_us - t_0_half_width_us
    t_0_max_us = t_0_echo_us + t_0_half_width_us
    t_0_vector_us = np.linspace(t_0_min_us, t_0_max_us, n_t_0s)

    # Calculate the x difference: x_element - x_sdh
    x_diff_array_mm = np.abs(x_gen_det_vector_mm - x_sdh_mm)

    z_sdh_m = ((0.5 * c_mpers * (t_0_vector_us * 10**-6)) +
               (radius_sdh_mm * 10**-3))

    # Pre-allocate output array:
    n_A_scans = len(x_gen_det_vector_mm)
    b_scan_ds_sdh_corrected = np.zeros((n_t_0s, n_A_scans))

    # Iterate over A-scans:
    for i, (A_scan_ds, x_diff_mm) in enumerate(zip(b_scan_ds.T,
                                                   x_diff_array_mm)):
        under_root = (x_diff_mm * 10**-3)**2 + z_sdh_m**2
        t_delays_s = (2/c_mpers) * (np.sqrt(under_root) -
                                    (radius_sdh_mm * 10**-3))
        b_scan_ds_sdh_corrected[:, i] = np.interp(t_delays_s,
                                                  time_vector_us * 10**-6,
                                                  A_scan_ds, 0, 0)

    return b_scan_ds_sdh_corrected, t_0_vector_us


def sdh_correct_XX_using_z_apex_vector(
    b_scan_ds,
    z_apex_vector_mm,
    time_vector_us,
    c_X_mpers
):
    """Correct for wave sets involving no mode conversion
    X-X = e.g. L-L or S-S.
    """

    radius_sdh_mm = params_al_sdh.radius_sdh_mm

    # Calculate z_sdh from z_apex by subtracting r_sdh:
    z_sdh_vector_m = (z_apex_vector_mm + radius_sdh_mm) * 10**-3

    # Calculate the x difference: x_element - x_sdh
    x_diff_array_mm = np.abs(params_al_sdh.x_offsets_from_sdh_mm)

    # Pre-allocate output array:
    n_z_apexs = len(z_apex_vector_mm)
    n_A_scans = params_al_sdh.n_tx
    b_scan_ds_sdh_corrected = np.zeros((n_z_apexs, n_A_scans))

    # Iterate over A-scans:
    for i, (A_scan_ds, x_diff_mm) in enumerate(zip(b_scan_ds.T,
                                                   x_diff_array_mm)):
        under_root = (x_diff_mm * 10**-3)**2 + z_sdh_vector_m**2
        t_delays_s = (2/c_X_mpers) * (np.sqrt(under_root) -
                                      (radius_sdh_mm * 10**-3))
        b_scan_ds_sdh_corrected[:, i] = np.interp(t_delays_s,
                                                  time_vector_us * 10**-6,
                                                  A_scan_ds, 0, 0)

    return b_scan_ds_sdh_corrected


def sdh_correct_HS_z_apex_based(
    b_scan_ds,
    z_apex_vector_mm,
    time_vector_us,
    c_lsaw_mpers
):
    # Fetch data from source for SDH experiment:
    c_s_mpers = material_al_sdh.c_T_mpers
    radius_sdh_mm = params_al_sdh.radius_sdh_mm

    # Generate a vector of z_sdh values to evaluate delay laws for:
    z_sdh_vector_m = (z_apex_vector_mm + radius_sdh_mm) * 10**-3

    # Calculate the x difference: x_element - x_sdh
    x_diff_array_m = np.abs(params_al_sdh.x_offsets_from_sdh_mm) * 10**-3

    # The total delay time can be expressed as:
    # t = (d_l/c_l) + (d_h/c_s) + (d_s/c_s)
    # Where d_l is the distance travelled as a LSAW,
    # d_h is the distance travelled as a shear head wave pre-reflection,
    # and d_s is the distance travelled as a shear wave on the return leg
    # post-reflection.

    # First calculate Δx*, the lateral distance between the centre of the SDH
    # (x_sdh) and the point on the surface where the head wave is born
    # (i.e. a ray passing through the centre of the sdh is at the
    # critical angle to the surface).
    delta_x_crit_vector_m = (material_al_sdh.tan_theta_crit *
                             z_sdh_vector_m)
    # Using Δx*, d_h can be calculated from a hypotenuse - r_SDH:
    d_h_vector_m = (np.sqrt(z_sdh_vector_m**2 + delta_x_crit_vector_m**2) -
                    (radius_sdh_mm * 10**-3))
    # d_l varies as a function of x_diff, so must be calculated in the loop.

    # Pre-allocate output array:
    n_z_apexs = len(z_apex_vector_mm)
    n_A_scans = params_al_sdh.n_tx
    b_scan_ds_sdh_hw_corrected = np.zeros((n_z_apexs, n_A_scans))

    # Iterate over A-scans:
    for i, (A_scan_ds, x_diff_m) in enumerate(zip(b_scan_ds.T,
                                                  x_diff_array_m)):
        # Calculate d_l, the distance travelled as an LSAW:
        d_l_vector_m = x_diff_m - delta_x_crit_vector_m
        # Calculate d_s, the distance travelled as an SV wave post-reflection:
        d_s_vector_m = (np.sqrt(z_sdh_vector_m**2 + x_diff_m**2) -
                        radius_sdh_mm * 10**-3)

        # Calculate total delays for 3-leg paths:
        t_delays_s = ((d_l_vector_m / c_lsaw_mpers) +
                      (d_h_vector_m / c_s_mpers) +
                      (d_s_vector_m / c_s_mpers))

        # Sample A-scan at delay times:
        b_scan_ds_sdh_hw_corrected[:, i] = np.interp(t_delays_s,
                                                     time_vector_us * 10**-6,
                                                     A_scan_ds, 0, 0)

    return b_scan_ds_sdh_hw_corrected


def sdh_correct_HS_bernspice(
    b_scan_ds_theo,
    z_apex_vector_mm,
    time_vector_theo_us
):
    """Moveout correction for the head wave signal in the Bernspice theoretical
    SDH B-scan dataset.

    The Bernstein & Spicer model cannot simulate reflection.
    To model the SDH B-scan dataset, the Bernspice model was evaluated at
    a radial position from the source with a distance of
    2 * the path length from the source to the edge of the SDH for each angle.
    This distance is equal to the total path length travelled for the L-L and
    S-S signals in reality, but does not accurately model the arrival time of
    the H-S signal as a function of gen/det position.

    The modelled head wave signals will arrive at a time t_HS_theo_s, where:
    t_HS = (d_LSAW / c_L) + (d_H / c_T)

    :param: b_scan_ds_theo: A-scans forming rows.
    """
    # The 'imaging grid' postulates a vector of possible values for z_apex,
    # the -axis depth of the SDH.
    # Convert the z_apex vector into a vector of possible values for the
    # SDH centre depth, z_centre, by adding the SDH radius:
    z_centre_vector_m = (z_apex_vector_mm +
                         params_al_sdh.radius_sdh_mm) * 10**-3

    x_offset_vector_m = params_al_sdh.x_offsets_from_sdh_mm * 10**-3

    # Now the calculation becomes angle-dependent.

    # Pre-allocate output array:
    n_z_apexs = len(z_apex_vector_mm)
    n_A_scans = params_al_sdh.n_tx
    corr_HS_theo = np.zeros((n_z_apexs, n_A_scans))

    # Loop over angle:
    for i, (A_scan_ds_theo, x_offset_m) in enumerate(zip(b_scan_ds_theo,
                                                         x_offset_vector_m)):
        # For this gen/det location, convert the vector of postulated
        # SDH depths into a vector of postulated total reflection path
        # lengths (r_m):
        r_source_to_circumference_m = (np.sqrt(z_centre_vector_m**2 +
                                               x_offset_m**2) -
                                       params_al_sdh.radius_sdh_mm * 10**-3)
        r_total_imaging_grid_m = 2 * r_source_to_circumference_m

        # Get the ray angles (theta) associated with these postulated
        # r_total values:
        # (angle measured from the surface normal
        # to the ray line that passes through the source and the centre of the SDH):
        theta_imaging_grid_deg = np.rad2deg(np.atan2(np.abs(x_offset_m),
                                                     z_centre_vector_m)
                                            )

        # Now, for this vector of postulated r_total distances, we
        # need to convert each into its associated postulated arrival time for
        # the head wave.

        # Resolve r_total into x and z components:
        x_r_imaging_grid_m = (r_total_imaging_grid_m *
                              np.sin(np.deg2rad(theta_imaging_grid_deg)))
        z_r_imaging_grid_m = (r_total_imaging_grid_m *
                              np.cos(np.deg2rad(theta_imaging_grid_deg)))

        # Calculate delta_x_crit, the lateral (x-axis) offset from the BernSpice evaluation
        # point r to the head wave birth point:
        delta_x_crit_imaging_grid_m = (material_al_sdh.tan_theta_crit *
                                       z_r_imaging_grid_m)

        # Subtract delta_x_crit from xr to find the path travelled by the simulated LSAW:
        d_lsaw_imaging_grid_m = (
            x_r_imaging_grid_m - delta_x_crit_imaging_grid_m)

        # Calculate the distance travelled by the simulated head wave, d_h,
        # using pythagoras:
        d_h_imaging_grid_m = np.sqrt(delta_x_crit_imaging_grid_m**2 +
                                     z_r_imaging_grid_m**2)

        # Divide distances by speeds and sum to get the estimated arrival time:
        t_arrival_imaging_grid_s = ((d_lsaw_imaging_grid_m / material_al_sdh.c_L_mpers) +
                                    (d_h_imaging_grid_m / material_al_sdh.c_T_mpers)
                                    )

        # Sample the A-scan for this iteration of the loop at the estimated
        # arrival times:
        corr_HS_theo[:, i] = np.interp(t_arrival_imaging_grid_s,
                                       time_vector_theo_us * 10**-6,
                                       A_scan_ds_theo,
                                       0,
                                       0)

    return corr_HS_theo
