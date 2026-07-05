import numpy as np


def get_xi(kappa, theta_array_rad):
    # kappa = c_l_mpers / c_t_mpers
    xi = np.emath.sqrt((1/kappa)**2 - np.sin(theta_array_rad)**2)
    return xi


def get_r_0(kappa, theta_array_rad):
    xi = get_xi(kappa, theta_array_rad)
    r_0 = np.cos(2 * theta_array_rad)**2 + (4 * (np.sin(theta_array_rad)**2) *
                                            xi * np.cos(theta_array_rad))
    return r_0


def get_generation_directivity_sv_riding(kappa, theta_array_rad):
    r_0 = get_r_0(kappa, theta_array_rad)
    g_sv = np.sin(4 * theta_array_rad) / r_0
    return g_sv


def get_detection_directivity_sv_riding(kappa, theta_array_rad):
    r_0 = get_r_0(kappa, theta_array_rad)
    xi = get_xi(kappa, theta_array_rad)
    d_sv = (2 * np.sin(2*theta_array_rad) * xi) / r_0
    return d_sv


def get_detection_directivity_l_pilant(kappa, theta_array_rad):
    # From W. Pilant (1979) 'Elastic waves in the earth', chapter 9, page 93.
    epsilon = (1 - ((2 / kappa**2) * np.sin(theta_array_rad)**2))
    d_l = ((np.cos(theta_array_rad) *
            epsilon
            ) /
           (epsilon**2 +
            ((4/kappa**3) *
             np.sin(theta_array_rad)**2 *
             np.cos(theta_array_rad) *
             np.emath.sqrt(1 - 
                           ((1/kappa**2) *
                            np.sin(theta_array_rad)**2
                            )
                           )
            )
            )
           )
    return d_l


def get_reflection_coefficient_sv_sv_riding(kappa, theta_in_array_rad):
    xi = get_xi(kappa, theta_in_array_rad)
    _A = np.cos(2 * theta_in_array_rad)**2
    _B = 2 * xi * np.sin(2 * theta_in_array_rad) * np.sin(theta_in_array_rad)
    numerator = _B - _A
    r_0 = _A + _B
    _r_sv_sv = - numerator / r_0
    return _r_sv_sv
