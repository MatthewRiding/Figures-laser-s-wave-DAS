# %%
"""Generate simulated A-scan sets using the Bernstein & Spicer model
for comparison against the experimental data from experiment 1:
the monostatic scan over the side-drilled hole (SDH).
"""
# %%
import numpy as np
from scipy.signal import hilbert

import bernstein_and_spicer_model as bernspice
from functions.moddirectivitiesriding import (
    get_detection_directivity_sv_riding,
    get_detection_directivity_l_pilant)
from functions.modfilterfmc3dbutter import filter_quartet
from corevariables.modsdhexperiment import (
    material_al_sdh,
    params_al_sdh)


def get_u_z_a_scan_at_angle_sdh(
    angle_theta_to_sdh_deg,
    path_length_m,
    time_vector_global_s
):
    """Get the A-scan of u_z out-of-plane surface displacement that
    the quartet interferometer would output at a given angle from the
    SDH.
    """
    # Calculate wave slownesses from wave speeds:
    s_l_sperm = 1 / material_al_sdh.c_L_mpers
    s_t_sperm = 1 / material_al_sdh.c_T_mpers

    # Convert angle from degrees to radians:
    angle_theta_to_sdh_rad = np.deg2rad(angle_theta_to_sdh_deg)

    # Find angle 'phi' in BernSpice coordinate system:
    phi_deg = bernspice.theta2phi(angle_theta_to_sdh_deg)

    # Get A-scans of displacement in polar coordinates for each wave:
    # Need to choose a radial position to evaluate at:
    # Choose 2 * the radial coordinate of the SDH for the chosen angle.
    r_m = path_length_m
    # WARNING: This will not compute the arrival time of the head wave
    # reflection H-S accurately, because it does not consider that
    # the post-reflection path of the HS signal is different to the
    # pre-reflection path.

    (u_r_due_to_l_wave_ascan_m,
     u_theta_due_to_direct_s_wave_ascan_m,
     u_theta_crit_due_to_head_s_wave_ascan_m
     ) = bernspice.get_bernspice_ascan_contributions_in_polar_coordinates(
         time_vector_global_s,
         r_m,
         phi_deg,
         s_l_sperm,
         s_t_sperm
    )

    # Bandpass filter to reflect quartet bandwidth:
    f_sampling_hz = 1 / (time_vector_global_s[1] - time_vector_global_s[0])
    (u_r_l_ascan_filtered_m
     ) = filter_quartet(u_r_due_to_l_wave_ascan_m,
                        f_sampling_hz)
    (u_theta_s_direct_ascan_filtered_m
     ) = filter_quartet(u_theta_due_to_direct_s_wave_ascan_m,
                        f_sampling_hz)
    (u_theta_crit_s_head_ascan_filtered_m
     ) = filter_quartet(u_theta_crit_due_to_head_s_wave_ascan_m,
                        f_sampling_hz)

    # u_z detection:

    # Compute complex values of the out-of-plane detection coefficients D
    # for the current angle:
    D_L_complex = get_detection_directivity_l_pilant(material_al_sdh.kappa,
                                                     angle_theta_to_sdh_rad)
    D_SV_complex = get_detection_directivity_sv_riding(material_al_sdh.kappa,
                                                       angle_theta_to_sdh_rad)

    # Multiply radial component due to longitudinal wave by
    # out-of-plane detection coefficient D_L to get u_z
    # component due to LL arrival:
    u_z_due_to_ll_ascan_m = np.real(
        hilbert(u_r_l_ascan_filtered_m) * D_L_complex)

    # Multiply theta component due to direct shear wave by
    # value of out-of-plane detection coefficient D_SV at
    # angle theta to get u_z component due to SS arrival:
    u_z_due_to_ss_ascan_m = np.real(
        hilbert(u_theta_s_direct_ascan_filtered_m) * D_SV_complex)

    # Multiply theta* component due to head shear wave by
    # value out-of-plane detection coefficient D_SV at angle
    # theta to get u_z component due to HS arrival:
    u_z_due_to_hs_ascan_m = np.real(
        hilbert(u_theta_crit_s_head_ascan_filtered_m) * D_SV_complex)

    # Sum the contibutions to the overall u_z A-scan from each wave reflection:
    u_z_total_ascan_m = (
        u_z_due_to_ll_ascan_m +
        u_z_due_to_ss_ascan_m +
        u_z_due_to_hs_ascan_m
    )

    # Filter to reflect the temporal freuency bandwidth of the quartet:
    u_z_quartet_ascan_m = u_z_total_ascan_m

    return u_z_quartet_ascan_m


def get_bernspice_a_scans_sdh(
    time_vector_global_s
):
    """Using data from the SDH experiment, generate a set of
    simulated A-scans equivalent to those captured experimentally.

    Parameters
    ----------
    time_vector_global_s : Vector of time points for which to evaluate
    measured displacements.

    Returns
    -------
    ndarray
        2D ndarray of shape (n_angles, n_times), where
        each row represents the u_z displacement A-scan recorded
        by the Quartet interferometer at one angle relative to the SDH.
    """
    # Compute the associated vector of distances from the SDH
    # based on the theta vector:

    # Get the list of angles for the A-scans captured during the experiment:
    # Can't use negative angles:
    theta_vector_deg = np.abs(params_al_sdh.angles_gen_sdh_deg)

    # Get the total path lengths of the reflected signals for each angle:
    (path_lengths_vector_m
     ) = params_al_sdh.direct_reflected_path_lengths_mm * 10**-3

    # Now loop over (theta, path_length) pairs & evaluate the displacement
    # A-scan observed at each (theta, r) coordinate point:
    # Time has been vectorised.
    # Pre-allocate output 'b-scan' array:
    n_elements = params_al_sdh.n_tx
    n_times = len(time_vector_global_s)
    u_z_ascans_sdh_bernspice_m = np.zeros((n_elements, n_times))

    for i, (theta_deg, path_length_m) in enumerate(zip(theta_vector_deg,
                                                       path_lengths_vector_m)):
        # Get out-of-plane displacement A-scan for this angle & store:
        (u_z_ascans_sdh_bernspice_m[i]
         ) = get_u_z_a_scan_at_angle_sdh(theta_deg,
                                         path_length_m,
                                         time_vector_global_s)

    return u_z_ascans_sdh_bernspice_m


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import colorcet as cc

    # Get set of modelled u_z A-scans for SDH experiment:

    # Generate a global time vector with a certain resolution:
    t_start_s = 0
    t_stop_s = 8 * 10**-6
    n_times = 4000
    time_vector_global_s = np.linspace(t_start_s, t_stop_s, n_times)

    u_z_ascans_sdh_bernspice_m = get_bernspice_a_scans_sdh(
        time_vector_global_s
        )

    # Plot displacements as a colour map (b-scan):

    # Image extent:
    extent = [np.min(params_al_sdh.x_elements_mm),
              np.max(params_al_sdh.x_elements_mm),
              np.max(time_vector_global_s) / 10**-6,
              np.min(time_vector_global_s) / 10**-6]

    # Color map:
    c_map = cc.m_CET_D7
    # Colormap symmetrical displacement limit:
    max_abs_m = 5 * 10**-9

    fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(5, 10))

    ax.imshow(u_z_ascans_sdh_bernspice_m.T,
              extent=extent,
              aspect='auto',
              vmin=-max_abs_m,
              vmax=max_abs_m,
              cmap=c_map)
    ax.set_xlabel('Monostatic scan position (mm)')
    ax.set_ylabel('Time (μs)')

    plt.show()
# %%
