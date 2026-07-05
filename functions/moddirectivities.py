import numpy as np


def directivity_thermoel_sv_scruby(kappa, theta_array_rad):
    # kappa = c_l_mpers / c_t_mpers
    G_numerator = kappa * np.sin(4 * theta_array_rad)
    G_factor_1 = kappa * (1 - (2.*np.sin(theta_array_rad)**2))**2
    G_factor_2 = 4 * (np.sin(theta_array_rad)**2) * \
        np.emath.sqrt((1 - np.sin(theta_array_rad)**2))
    G_factor_3 = np.emath.sqrt(1 - ((kappa**2)*np.sin(theta_array_rad)**2))
    G_denominator = G_factor_1 + (G_factor_2 * G_factor_3)
    directivity_sv_thermoel = G_numerator / G_denominator
    return directivity_sv_thermoel


def directivity_thermoel_sv_dankbaar(kappa, theta_array_rad):
    # Thermoelastic laser ultrasound in-plane dipole SV wave directivity
    # using Dankbaar notation on original equation from Scruby and Drain.
    # kappa = c_l_mpers / c_t_mpers
    xi = np.emath.sqrt((1/kappa**2) - np.sin(theta_array_rad)**2)
    f_0 = np.cos(2 * theta_array_rad)**2 + (2 * xi *
                                            np.sin(2 * theta_array_rad) *
                                            np.sin(theta_array_rad))
    G_SV = np.sin(4 * theta_array_rad) / f_0
    return G_SV
