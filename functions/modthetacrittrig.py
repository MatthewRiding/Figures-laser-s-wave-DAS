import numpy as np

from functions.modelasticityrelations import kappa_from_v_poisson


def tan_theta_crit(c_L_mpers, c_T_mpers):
    tan_theta_crit = c_T_mpers / np.sqrt(c_L_mpers**2 - c_T_mpers**2)
    return tan_theta_crit


def calculate_theta_crit_deg_kappa(kappa):
    # Kappa = c_L / c_T (See page 124 of Achenbach)
    theta_crit_deg = np.rad2deg(np.asin(1 / kappa))
    return theta_crit_deg


def calculate_theta_crit_deg(c_L_mpers, c_T_mpers):
    kappa = c_L_mpers / c_T_mpers
    theta_crit_deg = calculate_theta_crit_deg_kappa(kappa)
    return theta_crit_deg


def calculate_theta_crit_deg_poisson(v_poisson):
    kappa = kappa_from_v_poisson(v_poisson)
    theta_crit_deg = calculate_theta_crit_deg_kappa(kappa)
    return theta_crit_deg
