import numpy as np

from functions.modelasticityrelations import kappa_from_v_poisson


def calculate_sv_r_denominator(kappa, theta_in_array_rad,
                               theta_l_array_rad):
    denominator = ((np.sin(2 * theta_in_array_rad) * np.sin(2 *
                                                            theta_l_array_rad))
                   + ((kappa**2) * (np.cos(2 * theta_in_array_rad)**2)))
    return denominator


def calculate_theta_l(theta_in_array_rad, kappa):
    theta_l_array_rad = np.emath.arcsin(
        np.sin(theta_in_array_rad) * kappa)
    return theta_l_array_rad


def calculate_theta_s_from_theta_l(
    theta_l_incident_rad,
    kappa
):
    # kappa = c_l / c_s
    theta_s_reflected_rad = np.emath.arcsin((1 / kappa) *
                                            np.sin(theta_l_incident_rad))
    return theta_s_reflected_rad


def calculate_r_sv_l_auld(kappa, theta_in_array_rad):
    # Incident shear vertical wave (T) -> exiting longitudinal wave (L).
    theta_l_array_rad = calculate_theta_l(theta_in_array_rad, kappa)

    numerator = (2 * (kappa) * np.sin(2 * theta_in_array_rad) *
                 (np.cos(2 * theta_in_array_rad)))

    denominator = calculate_sv_r_denominator(kappa, theta_in_array_rad,
                                             theta_l_array_rad)

    r_sv_l = - (numerator/denominator)

    return r_sv_l


def calculate_r_sv_sv_auld(kappa, theta_in_array_rad):
    # Incident shear vertical wave (T) -> exiting shear vertical wave (T).
    # kappa = c_l_mpers / c_t_mpers
    theta_l_array_rad = calculate_theta_l(theta_in_array_rad, kappa)

    numerator = ((np.sin(2 * theta_in_array_rad) * np.sin(2 *
                                                          theta_l_array_rad))
                 - ((kappa**2) * (np.cos(2 * theta_in_array_rad)**2)))

    denominator = calculate_sv_r_denominator(kappa, theta_in_array_rad,
                                             theta_l_array_rad)

    r_sv_sv = - (numerator/denominator)

    return r_sv_sv


def calculate_r_l_l_auld(kappa, theta_l_incident_rad):
    """Auld 'Acoustic fields and waves and solids- vol II'
    page 32.
    Incident longitudinal wave -> exiting longitudinal wave.
    Reflection at a horizontal free surface.
    kappa = c_l / c_t
    Longitudinal wave incident at angle theta_l.
    Reflected mode converted SV wave exuant at angle theta_s.
    See figure 9.21 of Auld.
    """
    # Get angle of reflected mode-converted SV wave (theta_s):
    (theta_s_reflected_rad
     ) = calculate_theta_s_from_theta_l(theta_l_incident_rad,
                                        kappa)

    sines_term = (np.sin(2 * theta_s_reflected_rad) *
                  np.sin(2 * theta_l_incident_rad))
    cosine_term = (kappa**2 * np.cos(2 * theta_s_reflected_rad)**2)

    numerator = sines_term - cosine_term
    denominator = sines_term + cosine_term

    r_l_l = numerator / denominator

    return r_l_l


def calculate_r_sv_sv_auld_poisson(v_poisson, theta_in_array_rad):
    # Incident shear vertical wave (T) -> exiting shear vertical wave (T).
    kappa = kappa_from_v_poisson(v_poisson)

    r_sv_sv = calculate_r_sv_sv_auld(kappa, theta_in_array_rad)

    return r_sv_sv
