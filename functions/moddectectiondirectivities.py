import numpy as np

from functions.modthetacrittrig import calculate_theta_crit_deg_kappa


def a_n_over_b_0_scruby(kappa, theta_array_rad):
    # From Scruby & Drain 1990: Chapter 4.1, equation 4.14:
    # a_n is the normal component of displacement at a surface due to the
    # arrival of a shear vertical wave at an oblique angle theta.
    # B_0 is the shear displacement of the shear wave.
    # kappa = c_l_mpers / c_t_mpers
    theta_crit_rad = np.deg2rad(calculate_theta_crit_deg_kappa(kappa))

    a_n_over_b_0_subcrit = ((4 * np.sin(theta_array_rad) *
                            np.cos(theta_array_rad) *
                            np.emath.sqrt(1 - (kappa**2 *
                                               np.sin(theta_array_rad)**2)))
                            /
                            (
                            (4 * np.sin(theta_array_rad)**2 *
                             np.cos(theta_array_rad) *
                             np.emath.sqrt(1 - (kappa**2 *
                                                np.sin(theta_array_rad)**2))) +
                            (kappa * (1 - (2 * np.sin(theta_array_rad)**2))**2)
                            )
                            )

    a_n_over_b_0_supercrit = ((4 * np.sin(theta_array_rad) *
                               np.cos(theta_array_rad) *
                               np.emath.sqrt((kappa**2 *
                                              np.sin(theta_array_rad)**2) - 1))
                              /
                              np.emath.sqrt(
                                  (16 * np.sin(theta_array_rad)**4 *
                                   np.cos(theta_array_rad)**2 *
                                   ((kappa**2 * np.sin(theta_array_rad)**2)
                                    - 1)) +
                                  (kappa**2 *
                                   (1 - (2 * np.sin(theta_array_rad)**2))**4)
    )
    )

    a_n_over_b_0 = np.where(theta_array_rad <= theta_crit_rad,
                            a_n_over_b_0_subcrit,
                            a_n_over_b_0_supercrit)

    return a_n_over_b_0


def u_z_over_u_shear_dankbaar(c_l_mpers, c_t_mpers, theta_array_rad):
    # From Dankbaar (1985) 'Separation of P- and S-waves'.
    # A more complete derivation is given by Guevara in their Master's thesis:
    # 'Analysis and filtering of near-surface effects in land multicomponent
    # seismic data' (Appendix C: The free-surface effect).
    # Using the algebra of Dankbaar (1985):
    kappa = c_l_mpers / c_t_mpers

    # P is the 'ray parameter' or 'apparent slowness' on the surface
    # (see Guevara, pages 139 & 140):
    P = np.sin(theta_array_rad) / c_t_mpers
    epsilon = np.emath.sqrt((1/kappa)**2 - (c_t_mpers**2 * P**2))
    eta = np.emath.sqrt(1 - c_t_mpers**2 * P**2)
    R_0 = ((1 - (2 * c_t_mpers**2 * P**2))**2 +
           (4 * P**2 * c_t_mpers**2 * epsilon * eta)
           )

    R_shear_vertical = (4 * c_t_mpers * P * epsilon * eta) / R_0

    return R_shear_vertical
