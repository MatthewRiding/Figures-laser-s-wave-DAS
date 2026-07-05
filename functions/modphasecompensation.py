import numpy as np

from functions.moddirectivitiesriding import (get_generation_directivity_sv_riding,
                                              get_detection_directivity_sv_riding)
from functions.modreflectioncoefficients import calculate_r_sv_sv_auld


def compute_C_G_SV_from_rad(angles_gen_ray_rad, kappa):
    # kappa = c_L/c_S
    # For each angle in the array 'angles_gen_ray_rad',
    # compute the value of G_SV:
    G_SV = get_generation_directivity_sv_riding(kappa, angles_gen_ray_rad)
    # In order to counteract complex phase (and amplitude) modulating factor
    # G where:
    # G = a_G + ib_G = re^(i phi)
    # where:
    # r = sqrt(a_G^2 + b_G^2)
    # phi_G = arctan(b_G/a_G)
    # There exists unit compensation factor C where:
    # C = a_C + ib_C = 1 * e^(i phi_C)
    # phi_C = arctan(a_C + i_b_C) = - phi_G

    # Compute the phase angle of each G value (phi_G):
    phi_G_rad = np.angle(G_SV)
    # For each value of G, there is a corresponding C_G with unit amplitude
    # and phase angle - phi_G:
    C_G = np.exp(-1j * phi_G_rad)

    return C_G


def compute_C_G_SV_from_deg(angles_gen_ray_deg, kappa):
    # Convert input to radians:
    angles_gen_ray_rad = np.deg2rad(angles_gen_ray_deg)

    # Pass to radians version:
    C_G = compute_C_G_SV_from_rad(angles_gen_ray_rad, kappa)

    return C_G


def compute_C_D_SV_from_rad(angles_det_ray_rad, kappa):
    # kappa = c_L/c_S
    # For each angle in the array 'angles_det_ray_rad',
    # compute the value of D_SV:
    D_SV = get_detection_directivity_sv_riding(kappa, angles_det_ray_rad)
    # In order to counteract complex phase (and amplitude) modulating factor
    # D where:
    # D = a_D + ib_D = re^(i phi)
    # where:
    # r = sqrt(a_D^2 + b_D^2)
    # phi_D = arctan(b_D/a_D)
    # There exists unit compensation factor C where:
    # C = a_C + ib_C = 1 * e^(i phi_C)
    # phi_C = arctan(a_C + i_b_C) = - phi_D

    # Compute the phase angle of each D value (phi_D):
    phi_D_rad = np.angle(D_SV)
    # For each value of D, there is a corresponding C_D with unit amplitude
    # and phase angle - phi_D:
    C_D = np.exp(-1j * phi_D_rad)

    return C_D


def compute_C_D_SV_from_deg(angles_det_ray_deg, kappa):
    # Convert input to radians:
    angles_det_ray_rad = np.deg2rad(angles_det_ray_deg)

    # Pass to radians version:
    C_D = compute_C_D_SV_from_rad(angles_det_ray_rad, kappa)

    return C_D


def compute_C_R_SV_SV_from_rad(angles_incidence_rad, kappa):
    # kappa = c_L/c_S
    # For each angle in the array 'angles_incidence_rad',
    # compute the value of R_SV_SV:
    R_SV_SV = calculate_r_sv_sv_auld(kappa, angles_incidence_rad)
    # In order to counteract complex phase (and amplitude) modulating factor
    # R where:
    # R = a_R + ib_R = re^(i phi)
    # where:
    # r = sqrt(a_R^2 + b_R^2)
    # phi_R = arctan(b_R/a_R)
    # There exists unit compensation factor C where:
    # C = a_C + ib_C = 1 * e^(i phi_C)
    # phi_C = arctan(a_C + i_b_C) = - phi_R

    # Compute the phase angle of each R value (phi_R):
    phi_R_rad = np.angle(R_SV_SV)
    # For each value of R, there is a corresponding C_R with unit amplitude
    # and phase angle - phi_R:
    C_R = np.exp(-1j * phi_R_rad)

    return C_R


def compute_C_R_SV_SV_from_deg(angles_incidence_deg, kappa):
    # Convert input to radians:
    angles_incidence_rad = np.deg2rad(angles_incidence_deg)

    # Pass to radians version:
    C_R = compute_C_R_SV_SV_from_rad(angles_incidence_rad, kappa)

    return C_R


def get_C_90_hybridS(
    angles_spec_rays_vector_deg,
    theta_crit_deg
):
    """Compensate for the apparent +90 degree phase rotation of the head wave
    signal observed in experiment 2.

    C = 1 below theta_crit, then C = exp(-i * pi/2) above theta_crit.
    """
    # Pre-allocate an array of complex ones:
    C_90_hybrid_S = np.ones(len(angles_spec_rays_vector_deg),
                            dtype=np.complex128)

    # Define the -90 degree phase rotation factor:
    C_minus_90 = np.exp(-1j * np.pi/2)

    for index, angle_spec_ray_deg in enumerate(
        angles_spec_rays_vector_deg
    ):
        if angle_spec_ray_deg >= theta_crit_deg:
            C_90_hybrid_S[index] = C_minus_90
        # For sub-critical angles, C=1 + 0j.

    return C_90_hybrid_S
