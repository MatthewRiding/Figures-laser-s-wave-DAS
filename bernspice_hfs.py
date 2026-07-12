# %%
"""Generate simulated A-scan sets using the Bernstein & Spicer model
for comparison against the experimental data from experiment 2:
the normal move-out scan over the horizontal free surface (HFS).
"""
# %%
import numpy as np
from scipy.signal import hilbert

import bernstein_and_spicer_model as bernspice
from classdefs.modparamsets import material_param_set
from functions.moddirectivitiesriding import (
    get_detection_directivity_sv_riding,
    get_detection_directivity_l_pilant)
from functions.modreflectioncoefficients import (
    calculate_r_sv_sv_auld,
    calculate_r_l_l_auld
)
from corevariables.modhfsexperiment import params_al_nmo
from functions.modfilterfmc3dbutter import filter_quartet


material_al_hfs = material_param_set(c_L_mpers=6371,
                                     c_T_mpers=3077)


def get_u_z_a_scan_at_angle_hfs(
    angle_theta_nmo_deg,
    path_length_nmo_m,
    time_vector_global_s
):
    """Get the A-scan of u_z out-of-plane surface displacement that
    the quartet interferometer would output for a given NMO angle
    on the 2mm-thick HFS sample.
    """
    # Calculate wave slownesses from wave speeds:
    s_l_sperm = 1 / material_al_hfs.c_L_mpers
    s_t_sperm = 1 / material_al_hfs.c_T_mpers

    # Convert angle from degrees to radians:
    angle_theta_nmo_rad = np.deg2rad(angle_theta_nmo_deg)

    # Find angle 'phi' in BernSpice coordinate system:
    phi_nmo_deg = bernspice.theta2phi(angle_theta_nmo_deg)

    # Get A-scans of displacement in polar coordinates for each wave:
    # Need to choose a radial position to evaluate at.
    # Choose the path length of the NMO reflection for this angle:
    r_m = path_length_nmo_m

    (u_r_due_to_l_wave_ascan_m,
     u_theta_due_to_direct_s_wave_ascan_m,
     u_theta_crit_due_to_head_s_wave_ascan_m
     ) = bernspice.get_bernspice_ascan_contributions_in_polar_coordinates(
         time_vector_global_s,
         r_m,
         phi_nmo_deg,
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

    # Multiplying by reflection and detection directivity patterns:
    # We will NOT consider the head wave generated on the bottom surface
    # by mode conversion from the L-wave.

    # Reflection coefficients:
    # Compute L-L and SV-SV reflection coefficients:
    r_l_l_complex = calculate_r_l_l_auld(material_al_hfs.kappa,
                                         angle_theta_nmo_rad)
    r_sv_sv_complex = calculate_r_sv_sv_auld(material_al_hfs.kappa,
                                             angle_theta_nmo_rad)
    r_sv_sv_crit_complex = calculate_r_sv_sv_auld(
        material_al_hfs.kappa,
        np.deg2rad(material_al_hfs.theta_crit_deg))

    # Compute complex values of the out-of-plane detection coefficients D
    # for the current angle & critical angle (for the head wave):
    D_L_theta_complex = get_detection_directivity_l_pilant(
        material_al_hfs.kappa,
        angle_theta_nmo_rad)
    D_SV_theta_complex = get_detection_directivity_sv_riding(
        material_al_hfs.kappa,
        angle_theta_nmo_rad)
    D_SV_crit_complex = get_detection_directivity_sv_riding(
        material_al_hfs.kappa,
        np.deg2rad(material_al_hfs.theta_crit_deg))

    # Multiply radial component due to longitudinal wave by:
    # 1. L-L reflection coefficient R_L_L
    # 2. Out-of-plane detection coefficient D_L to get u_z
    # component due to LL arrival:
    u_z_due_to_ll_ascan_m = np.real(
        hilbert(u_r_l_ascan_filtered_m) * r_l_l_complex * D_L_theta_complex)

    # Multiply theta component due to direct shear wave by:
    # 1. The reflection coefficient R_SV_SV at the NMO angle.
    # 2. The value of out-of-plane detection coefficient D_SV at
    # angle theta to get u_z component due to SS arrival:
    u_z_due_to_ss_ascan_m = np.real(
        hilbert(u_theta_s_direct_ascan_filtered_m) *
        r_sv_sv_complex *
        D_SV_theta_complex)

    # Multiply theta* component due to head shear wave by:
    # 1. The value of the reflection coefficient R_SV_SV at the critical angle.
    # 2. The value out-of-plane detection coefficient D_SV at the
    # critical angle, to get u_z component due to HS arrival:
    u_z_due_to_hs_ascan_m = np.real(
        hilbert(u_theta_crit_s_head_ascan_filtered_m) *
        r_sv_sv_crit_complex *
        D_SV_crit_complex)

    # Sum the contibutions to the overall u_z A-scan from each wave reflection:
    # Multiply by -1 to reflect the change in direction of the waves:
    u_z_total_ascan_m = -1 * (
        u_z_due_to_ll_ascan_m +
        u_z_due_to_ss_ascan_m +
        u_z_due_to_hs_ascan_m
    )

    # Filter to reflect the temporal freuency bandwidth of the quartet:
    u_z_quartet_ascan_m = u_z_total_ascan_m

    return u_z_quartet_ascan_m


def get_bernspice_a_scans_hfs(
    time_vector_global_s
):
    """Using data from the HFS experiment, generate a set of
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
        by the Quartet interferometer at one NMO reflection angle.
    """
    # Compute the associated vector of NMO path lengths
    # based on the theta vector:
    theta_vector_deg = params_al_nmo.theta_vector_deg
    path_lengths_vector_m = params_al_nmo.path_lengths_vector_mm * 10**-3

    # Now loop over (theta, path_length) pairs:
    # Time has been vectorised.
    # Pre-allocate output 'b-scan' array:
    n_elements = params_al_nmo.n_elements
    n_times = len(time_vector_global_s)
    u_z_ascans_hfs_bernspice_m = np.zeros((n_elements, n_times))

    for i, (theta_deg, path_length_m) in enumerate(zip(theta_vector_deg,
                                                       path_lengths_vector_m)):
        # Get out-of-plane displacement A-scan for this angle & store:
        (u_z_ascans_hfs_bernspice_m[i]
         ) = get_u_z_a_scan_at_angle_hfs(theta_deg,
                                         path_length_m,
                                         time_vector_global_s)

    return u_z_ascans_hfs_bernspice_m


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import colorcet as cc

    # Get modelled u_z A-scans for experiment:

    # Generate a global time vector with a certain resolution:
    t_start_s = 0
    t_stop_s = 4 * 10**-6
    n_times = 4000
    time_vector_global_s = np.linspace(t_start_s, t_stop_s, n_times)

    (u_z_ascans_hfs_bernspice_m
     ) = get_bernspice_a_scans_hfs(time_vector_global_s)

    # Plot displacements as a colour map (b-scan):

    # Image extent:
    extent = [np.min(params_al_nmo.x_vector_mm),
              np.max(params_al_nmo.x_vector_mm),
              np.max(time_vector_global_s) / 10**-6,
              np.min(time_vector_global_s) / 10**-6]

    # Color map:
    c_map = cc.m_CET_D7
    # Colormap symmetrical limit:
    max_abs_m = 5 * 10**-9

    fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(5, 10))

    ax.imshow(u_z_ascans_hfs_bernspice_m.T,
              extent=extent,
              aspect='auto',
              vmin=-max_abs_m,
              vmax=max_abs_m,
              cmap=c_map)
    ax.set_xlabel('NMO surface offset (mm)')
    ax.set_ylabel('Time (μs)')

    plt.show()
