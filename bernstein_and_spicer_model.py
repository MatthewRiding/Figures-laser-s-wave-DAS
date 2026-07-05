# %%
"""
Principle reference: Bernstein and Spicer (2000):
'Line source representation for laser-generated ultrasound in aluminum'

Full citation:
Johanna R. Bernstein, James B. Spicer;
Line source representation for laser-generated ultrasound in aluminum.
J. Acoust. Soc. Am. 1 March 2000; 107 (3): 1352–1357.
DOI: https://doi.org/10.1121/1.428422


ABSOLUTELY CRUCIAL:
BERNSTEIN AND SPICER (2000) use the following coordinate system:
y-direction: positive = downwards into sample, normal to surface.
x-direction: parallel to surface, perpendicular to laser line.
Positive = away from source.
z-direction: parallel to surface, parallel to laser line.
phi: angle between ray and surface
(NOT ANGLE BETWEEN RAY AND SURFACE NORMAL).

Errors in print:
The equations for this model can be found in the above published
journal article [Bernstein et al. (2000)], and in the PhD thesis
of J. R. Bernstein: "ULTRASONIC CRACK DIFFRACTION IN METALS:
INVESTIGATIONS USING LASER GENERATED ULTRASONIC SHEAR WAVES
AND BROADBAND EMAT DETECTION" (The Johns Hopkins University, 1999).
In both published formats, there are errors in the formulae written
in the body text.  However, the original MATLAB scripts used by
J. R. Bernstein to run their model are given as an appendix to the
PhD thesis.  These scripts contain the correct formulae, and
were used as the basis for this python implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc


# %% Model equations:


def get_gamma_l(eta, s_l_sperm):
    """Equation (7a) of Bernstein & Spicer (2000).

    Parameters
    ----------
    eta :
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.

    Returns
    -------
    """
    gamma_l = np.emath.sqrt(s_l_sperm**2 - eta**2)
    return gamma_l


def get_gamma_t(eta, s_t_sperm):
    """Equation (7b) of Bernstein & Spicer (2000).

    Parameters
    ----------
    eta :
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    """
    gamma_t = np.emath.sqrt(s_t_sperm**2 - eta**2)
    return gamma_t


def get_rayleigh_function(eta, s_l_sperm, s_t_sperm):
    """Equation (6) of Bernstein & Spicer (2000).

    Parameters
    ----------
    eta : Eta parameter.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    """
    gamma_l = get_gamma_l(eta, s_l_sperm)
    gamma_t = get_gamma_t(eta, s_t_sperm)
    rayleigh_of_eta = (
        (4 * gamma_l * gamma_t * eta**2) +
        (s_t_sperm**2 - (2 * eta**2))**2
    )
    return rayleigh_of_eta


def get_eta_t_plus(t_s, r_m, phi_deg, s_t_sperm):
    """Equation (10b) of Bernstein & Spicer (2000), choosing the
    'addition' operator at the plus/minus sign.

    Parameters
    ----------
    t_s : Evaluation time. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    """
    eta_t_plus = ((- (t_s / r_m) * np.cos(np.deg2rad(phi_deg))) +
                  (1j * np.sin(np.deg2rad(phi_deg)) *
                   np.emath.sqrt((t_s**2 / r_m**2) - s_t_sperm**2)
                   )
                  )
    return eta_t_plus


def get_d_eta_t_plus_d_t(t_s, r_m, phi_deg, s_t_sperm):
    """Equation for the partial derivative of equation (10b) from
    Bernstein & Spicer (2000) with respect to time.
    Used in equations (12a) and (12b) of the paper.

    Parameters
    ----------
    t_s : Evaluation time. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    """
    d_eta_plus_d_t = (
        (-np.cos(np.deg2rad(phi_deg)) / r_m) +
        ((1j * np.sin(np.deg2rad(phi_deg))) *
         (t_s /
          (r_m**2 * np.emath.sqrt((t_s**2 / r_m**2) -
                                  s_t_sperm**2)
           )
          )
         )
    )
    return d_eta_plus_d_t


def get_eta_tl(t_s, r_m, phi_deg, s_t_sperm):
    """Ommitted from the journal article Bernstein and Spicer (2000), but
    required for the evaluation of equations (14a) and (14b) of the paper.
    The formula for eta_TL is given in the PhD thesis of J. R. Bernstein,
    on page 27, as equation (2.20).  Also given in thesis appendix C:
    MATLAB programs (page 139).

    Parameters
    ----------
    t_s : Evaluation time. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    """
    eta_tl = (
        ((-t_s / r_m) * np.cos(np.deg2rad(phi_deg))) +
        (np.sin(np.deg2rad(phi_deg)) *
         np.emath.sqrt((s_t_sperm**2 - (t_s**2 / r_m**2))))
    )
    return eta_tl


def get_d_eta_tl_dt(t_s, r_m, phi_deg, s_t_sperm):
    """Partial derivative of eta_TL with respect to time.
    Required to evaluate equations (14a) and (14b) from the
    journal article.

    Parameters
    ----------
    t_s : Evaluation time. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    _type_
        _description_
    """
    d_eta_tl_dt = (
        (-np.cos(np.deg2rad(phi_deg)) / r_m) -
        (np.sin(np.deg2rad(phi_deg)) *
         (t_s / (r_m**2 * np.emath.sqrt(s_t_sperm**2 - (t_s**2 / r_m**2)))
          )
         )
    )
    return d_eta_tl_dt


def get_eta_l_plus(t_s, r_m, phi_deg, s_l_sperm):
    """Equation (10a) of Bernstein & Spicer (2000), choosing the
    'addition' operator at the plus/minus sign.

    Parameters
    ----------
    t_s : Evaluation time. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.

    Returns
    -------
    """
    eta_l_plus = (
        (- (t_s / r_m) *
         np.cos(np.deg2rad(phi_deg))
         ) +
        (1j *
         np.sin(np.deg2rad(phi_deg)) *
         np.emath.sqrt((t_s**2 / r_m**2) - s_l_sperm**2)
         )
    )
    return eta_l_plus


def get_d_eta_l_plus_dt(t_s, r_m, phi_deg, s_l_sperm):
    """Partial derivative of equation (10a) of Bernstein & Spicer (2000),
    (choosing the 'addition' operator at the plus/minus sign) with respect
    to time.

    Parameters
    ----------
    t_s : Evaluation time. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.

    Returns
    -------
    """
    d_eta_l_plus_d_t = (
        (-np.cos(np.deg2rad(phi_deg)) /
         r_m
         ) +
        (1j *
         np.sin(np.deg2rad(phi_deg)) *
         (t_s /
          (r_m**2 *
           np.emath.sqrt(
               (t_s**2 / r_m**2) -
               s_l_sperm**2)
           )
          )
         )
    )
    return d_eta_l_plus_d_t


def get_t_tl(r_m, phi_deg, s_l_sperm, s_t_sperm):
    """Equation (13) of Bernstein & Spicer (2000).
    Note error in Bernstein & Spicer paper (2000) & body text of Bernstein's
    PhD thesis.
    Error: Missing r factor in first term.
    The r is included correctly in the MATLAB scripts appended Bernstein's
    thesis.

    Parameters
    ----------
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    """
    t_tl_s = ((r_m * s_l_sperm * np.cos(np.deg2rad(phi_deg))) +
              (r_m * np.sin(np.deg2rad(phi_deg)) *
               np.emath.sqrt(s_t_sperm**2 - s_l_sperm**2)
               )
              )
    return t_tl_s


def get_heaviside_st(time_vector_s, r_m, s_t_sperm):
    """Heaviside factor used in equations (12a), (12b),
    (14a) and (14b) of Bernstein & Spicer (2000) denoted H(t - s_Tr).

    Parameters
    ----------
    time_vector_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    Heaviside step function: 1 where time_vector_s >= (s_t_sperm * r_m),
    else 0.
    """
    heaviside_vector = np.where(time_vector_s >= (s_t_sperm * r_m), 1, 0)
    return heaviside_vector


def get_heaviside_sl(t_s, r_m, s_l_sperm):
    """Heaviside factor used in equations (11a) & (11b) of Bernstein & Spicer
    (2000) denoted H(t - s_Lr).  Note: error in equation (11b) of journal
    article: H(t - s_Tr) should be H(t - s_Lr).  Equivalent equations in
    J. R. Bernstein thesis (2.18a,b) are correct, as are those used in the
    MATLAB scripts (appendix C of thesis).

    Parameters
    ----------
    time_vector_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.

    Returns
    -------
    Heaviside step function: 1 where time_vector_s >= (s_l_sperm * r_m),
    else 0.
    """
    heaviside_factor = np.where(t_s >= (s_l_sperm * r_m), 1, 0)
    return heaviside_factor


def get_heaviside_tl(time_vector_s, r_m, phi_deg, s_l_sperm, s_t_sperm):
    """Heaviside factor used in equations (14a) and (14b) of
    Bernstein & Spicer (2000) denoted H(t - t_TL).

    Parameters
    ----------
    time_vector_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    Heaviside step function: 1 where time_vector_s >= t_TL,
    else 0.
    """
    t_tl_s = get_t_tl(r_m, phi_deg, s_l_sperm, s_t_sperm)
    heaviside_vector = np.where(time_vector_s >= t_tl_s, 1, 0)
    return heaviside_vector


def get_h_tl_minus_h_st(time_vector_s, r_m, phi_deg, s_l_sperm, s_t_sperm):
    """Difference of Heaviside factors used in equations (14a) and (14b) of
    Bernstein & Spicer (2000).

    Parameters
    ----------
    time_vector_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    """
    h_vector_tl = get_heaviside_tl(time_vector_s, r_m, phi_deg,
                                   s_l_sperm, s_t_sperm)
    h_vector_st = get_heaviside_st(time_vector_s, r_m, s_t_sperm)
    h_vector_tl_minus_h_st = h_vector_tl - h_vector_st
    return h_vector_tl_minus_h_st


def get_amplitude_constant():
    """Amplitude constant denoted A/(mu * pi) in the algebra of Bernstein &
    Spicer (2000).
    Used in equations (11a), (11b), (12a), (12b), (14a), (14b) of the journal
    article.

    Returns
    -------
    A / (mu * pi) using a hard-coded value of the shear modulus mu.
    """
    # A: a constant amplitude factor related to the pulse energy
    # and the thermal & elastic properties of the material.
    A = 1

    # Mu: Lame's second paramter, or shear modulus for material.
    # Using approximate value for Aluminium: 26 GPa
    mu_pa = 26 * 10**9

    constant = A / (mu_pa * np.pi)
    return constant


def theta2phi(theta_deg):
    """Converts the angle from the surface normal (theta, in degrees)
    to the associated angle from the surface (phi, in degrees).
    Bernstein and Spicer wrote their model in terms of phi.
    Accepts theta in degrees and returns phi in units of degrees.
    """
    phi_deg = 90 - theta_deg
    return phi_deg


def get_u_t_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm):
    """Equation (12a) of Bernstein & Spicer (2000).

    Parameters
    ----------
    t_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    Component of displacement in the x-axis direction (parallel to
    sample surface, see figure 1 of journal article) at coordinate (phi, r)
    at time t due to the direct transverse wave.
    """

    constant = get_amplitude_constant()

    eta_t_plus = get_eta_t_plus(t_s, r_m, phi_deg, s_t_sperm)

    gamma_t = get_gamma_t(eta_t_plus, s_t_sperm)

    term_in_eta_numerator = (eta_t_plus *
                             gamma_t *
                             (s_t_sperm**2 - (2 * eta_t_plus**2)))

    rayleigh_of_eta = get_rayleigh_function(eta_t_plus, s_l_sperm, s_t_sperm)

    d_eta_plus_d_t = get_d_eta_t_plus_d_t(t_s, r_m, phi_deg, s_t_sperm)

    heaviside_term = get_heaviside_st(t_s, r_m, s_t_sperm)

    u_t_m = (-constant *
             np.imag((term_in_eta_numerator / rayleigh_of_eta) *
                     d_eta_plus_d_t) *
             heaviside_term)

    return u_t_m


def get_v_t_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm):
    """Equation (12b) of Bernstein & Spicer (2000).

    Parameters
    ----------
    t_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    Component of displacement in the y-axis direction (normal to
    sample surface, see figure 1 of journal article) at coordinate (phi, r)
    at time t due to the direct transverse wave.
    """

    constant = get_amplitude_constant()

    eta_t_plus = get_eta_t_plus(t_s, r_m, phi_deg, s_t_sperm)

    term_in_eta_numerator = (eta_t_plus**2 *
                             (s_t_sperm**2 - (2 * eta_t_plus**2)))

    rayleight_of_eta = get_rayleigh_function(eta_t_plus, s_l_sperm, s_t_sperm)

    d_eta_plus_d_t = get_d_eta_t_plus_d_t(t_s, r_m, phi_deg, s_t_sperm)

    heaviside_term = get_heaviside_st(t_s, r_m, s_t_sperm)

    v_t_m = (-constant *
             np.imag((term_in_eta_numerator /
                      rayleight_of_eta) *
                     d_eta_plus_d_t) *
             heaviside_term)

    return v_t_m


def resolve_u_theta_from_bernstein_u_and_v(u_m, v_m, phi_deg):
    """Resolves cartesian displacement components from cartesian (x,y) system
    used by Bernstein & Spicer (2000) onto a circular polar coordinate system
    (theta, r) centred on the source point, and returns the component of
    displacement in the theta direction.

    - At phi=90 deg (surface normal), v (the displacement in the y axis)
    contributes nothing to shear in the angular direction.
    - At phi=0 deg (surface parallel), u (the displacement in the x-axis)
    contributes nothing to shear in the angular direction.
    u contributes with sin(phi) dependence.
    v contibutes with cos(phi) dependence.
    These components pull in opposite directions (i.e. have opposite sign)
    on the new angled axis that points from the surface normal to the
    surface parallel.

    Parameters
    ----------
    u_m : Component of displacement in the x-axis.  Units: metres.
    v_m : Component of displacement in the y-axis.  Units: metres.
    phi_deg : Angle from the surface of new axis to resolve displacement
    component onto.

    Returns
    -------
    Component of displacement on axis orientated at angle phi to the surface
    (angle theta to the surface normal).
    """
    # Resolve components onto angled 'theta' axis:
    u_theta_m = ((u_m * np.sin(np.deg2rad(phi_deg))) -
                 (v_m * np.cos(np.deg2rad(phi_deg)))
                 )
    return u_theta_m


def resolve_u_r_from_bernstein_u_and_v(u_m, v_m, phi_deg):
    """Resolves cartesian displacement components from cartesian (x,y) system
    used by Bernstein & Spicer (2000) onto a circular polar coordinate system
    (theta, r) centred on the source point, and returns the component of
    displacement in the radial direction.

    - At phi=90 deg (surface normal), u (the displacement in the x-axis)
    contributes nothing to radial displacement.
    - At phi=0 deg (surface parallel), v (the displacement in the y-axis)
    contributes nothing to radial displacement.
    u contributes with cos(phi) dependence.
    v contibutes with sin(phi) dependence.
    These components pull in opposite directions (i.e. have opposite sign)
    on the new angled axis that points from the surface normal to the
    surface parallel.

    Parameters
    ----------
    u_m : Component of displacement in the x-axis.  Units: metres.
    v_m : Component of displacement in the y-axis.  Units: metres.
    phi_deg : Angle from the surface of new axis to resolve displacement
    component onto.

    Returns
    -------
    Component of displacement on axis orientated at angle phi to the surface
    (angle theta to the surface normal).
    """
    u_r_m = ((u_m * np.cos(np.deg2rad(phi_deg))) +
             (v_m * np.sin(np.deg2rad(phi_deg)))
             )
    return u_r_m


def resolve_u_theta_crit_from_bernstein_u_and_v(
    u_m,
    v_m,
    c_l_mpers,
    c_t_mpers
):
    """u_theta_crit is the component of displacement in the direction of
    an axis that intersects the surface of the sample at the critical angle
    theta_crit.  For example, the shear head wave creates displacements
    parallel to this axis only.
    """
    cos_theta_crit = np.sqrt(1 - (c_t_mpers**2 / c_l_mpers**2))
    sin_theta_crit = c_t_mpers / c_l_mpers

    u_theta_crit = ((u_m * cos_theta_crit) -
                    (v_m * sin_theta_crit)
                    )
    return u_theta_crit


def get_u_theta_t_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm):
    """Get component of displacement in angular (theta) direction
    at coordinate point (phi, r) at time t.

    Parameters
    ----------
    t_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    Component of displacement in theta direction.  Units: metres.
    """
    # First, I'll compute the components of displacement in
    # Bernstein & Spicer's (u, v) coordinae system:
    # Direct shear wave:
    u_t_m = get_u_t_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm)
    v_t_m = get_v_t_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm)

    # Resolve components onto angled 'theta' axis:
    u_theta_t_m = resolve_u_theta_from_bernstein_u_and_v(u_t_m, v_t_m, phi_deg)
    return u_theta_t_m


def get_u_tl_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm):
    """Equation 14(a) from Bernstein & Spicer (2000).
    Returns the component of displacement in the x-axis sirection
    at coordinate point (phi, r) at time t due to the shear head wave.

    Parameters
    ----------
    t_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    Component of displacement in x-axis direction.  Units: metres.
    """
    # Equation 14(a) of Bernstein & Spicer (2000):
    constant = get_amplitude_constant()

    eta_tl = get_eta_tl(t_s, r_m, phi_deg, s_t_sperm)

    gamma_t_tl = get_gamma_t(eta_tl, s_t_sperm)

    rayleigh_of_eta_tl = get_rayleigh_function(eta_tl, s_l_sperm, s_t_sperm)

    d_eta_tl_dt = get_d_eta_tl_dt(t_s, r_m, phi_deg, s_t_sperm)

    heaviside_term_tl = get_heaviside_tl(t_s, r_m, phi_deg,
                                         s_l_sperm, s_t_sperm)
    heaviside_term_st = get_heaviside_st(t_s, r_m, s_t_sperm)

    h_tl_minus_h_st = heaviside_term_tl - heaviside_term_st

    term_in_eta_numerator = (
        eta_tl * gamma_t_tl * (s_t_sperm**2 - (2 * eta_tl**2))
    )

    u_tl_m = (
        (- constant) *
        np.imag(
            (term_in_eta_numerator / rayleigh_of_eta_tl) *
            d_eta_tl_dt
        ) *
        h_tl_minus_h_st
    )

    return u_tl_m


def get_v_tl_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm):
    """Equation 14(b) from Bernstein & Spicer (2000).
    Returns the component of displacement in the y-axis sirection
    at coordinate point (phi, r) at time t due to the shear head wave.

    Parameters
    ----------
    t_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    Component of displacement in y-axis direction.  Units: metres.
    """
    # Equation 14(b) of Bernstein & Spicer (2000):
    constant = get_amplitude_constant()

    eta_tl = get_eta_tl(t_s, r_m, phi_deg, s_t_sperm)

    rayleigh_of_eta_tl = get_rayleigh_function(eta_tl, s_l_sperm, s_t_sperm)

    d_eta_tl_dt = get_d_eta_tl_dt(t_s, r_m, phi_deg, s_t_sperm)

    heaviside_term_tl = get_heaviside_tl(t_s, r_m, phi_deg,
                                         s_l_sperm, s_t_sperm)
    heaviside_term_st = get_heaviside_st(t_s, r_m, s_t_sperm)

    h_tl_minus_h_st = heaviside_term_tl - heaviside_term_st

    term_in_eta_numerator = (
        eta_tl**2 * (s_t_sperm**2 - (2 * eta_tl**2))
    )

    v_tl_m = (
        (- constant) *
        np.imag(
            (term_in_eta_numerator / rayleigh_of_eta_tl) *
            d_eta_tl_dt
        ) *
        h_tl_minus_h_st
    )

    return v_tl_m


def get_u_theta_tl_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm):
    """Get component of displacement in the theta direction at
    coordinate point (phi, r) at time t due to the shear head wave.

    Parameters
    ----------
    t_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    Component of displacement in theta direction.  Units: metres.
    """
    # Get displacement in polar theta direction due to transverse head wave.

    # Get displacement components in original
    # cartesian coordinate system used by Bernstein:
    u_tl_m = get_u_tl_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm)
    v_tl_m = get_v_tl_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm)

    # Resolve theta component in (r, theta) polar coordinates:
    u_theta_tl_m = resolve_u_theta_from_bernstein_u_and_v(u_tl_m,
                                                          v_tl_m,
                                                          phi_deg)
    return u_theta_tl_m


def get_u_l_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm):
    """Equation (11a) of Bernstein and Spicer (2000).
    Component of displacement in x-axis direction at coordinate
    (phi, r) at time t due to longitudinal wave.

    Parameters
    ----------
    t_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    Component of displacement in x-axis direction. Units: metres.
    """
    constant = get_amplitude_constant()
    eta_l_plus = get_eta_l_plus(t_s, r_m, phi_deg, s_l_sperm)
    gamma_t = get_gamma_t(eta_l_plus, s_t_sperm)
    rayleigh_of_eta = get_rayleigh_function(eta_l_plus, s_l_sperm, s_t_sperm)
    d_eta_l_plus_dt = get_d_eta_l_plus_dt(t_s, r_m, phi_deg, s_l_sperm)
    heaviside_factor = get_heaviside_sl(t_s, r_m, s_l_sperm)

    numerator_in_eta = (2 *
                        eta_l_plus**3 *
                        gamma_t)

    u_l_m = (constant *
             np.imag(
                 (numerator_in_eta / rayleigh_of_eta) *
                 d_eta_l_plus_dt
             ) *
             heaviside_factor
             )

    return u_l_m


def get_v_l_m(t_s, r_m, phi_deg, s_l_sperm, s_t_sperm):
    """Equation (11b) of Bernstein and Spicer (2000).
    Component of displacement in y-axis direction at coordinate
    (phi, r) at time t due to longitudinal wave.

    Note: Three errors in algebra presented in the published JASA article.
    2 errors not present in J. Bernstein thesis (equation 2.18b).
    - Fraction over rayleigh function incorrectly denoted as
    evaluated at eta_{T+}.  Should be evaluated at eta_{L+}.
    - Heaviside term notated as evalauted at (t - s_T * r).
    Should be evaluated at (t - s_L * r).
    1 error reproduced in equation 2.18b of J. Bernstein thesis, but not
    present in appended MATLAB scripts.
    - Partial derivative of eta with respect to time denoted as
    d_eta_{T+}/dt, when it should be d_eta_{L+}/dt.

    Parameters
    ----------
    t_s : Evaluation time or times. Units: Seconds.
    r_m : Evaluation radial distance from source. Units: metres.
    phi_deg : Evaluation angle from surface.  Units: Degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    Component of displacement in y-axis direction. Units: metres.
    """
    constant = get_amplitude_constant()
    eta_l_plus = get_eta_l_plus(t_s, r_m, phi_deg, s_l_sperm)
    gamma_l = get_gamma_l(eta_l_plus, s_l_sperm)
    gamma_t = get_gamma_t(eta_l_plus, s_t_sperm)
    rayleigh_of_eta = get_rayleigh_function(eta_l_plus, s_l_sperm, s_t_sperm)
    d_eta_l_plus_dt = get_d_eta_l_plus_dt(t_s, r_m, phi_deg, s_l_sperm)
    heaviside_factor = get_heaviside_sl(t_s, r_m, s_l_sperm)

    numerator_in_eta = (2 *
                        eta_l_plus**2 *
                        gamma_l *
                        gamma_t
                        )

    v_l_m = (- constant *
             np.imag(
                 (numerator_in_eta / rayleigh_of_eta) *
                 d_eta_l_plus_dt
             ) *
             heaviside_factor
             )

    return v_l_m


def get_u_theta_a_scans_at_angle(
    theta_deg,
    radius_observation_m,
    time_vector_s,
    s_l_sperm,
    s_t_sperm,
):
    """Get the time traces (A-scans) of the component of displacement
    in the theta direction at the coordinate point (theta, r) due to
    the direct & head shear waves separately.

    Parameters
    ----------
    theta_deg : Angle from surface normal.  Units: Degrees.
    radius_observation_m : Radial coordinate from the source point.
        Units: Metres.
    time_vector_s : Vector of time values at which to evaluate the
        shear displacement.  Units: seconds
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    tuple
    (u_theta_t_a_scan_m, u_theta_tl_a_scan_m)
    Vectors of displacement.  Units: Metres.
    """
    # Convert theta to phi:
    phi_deg = theta2phi(theta_deg)

    # Compute polar displacement due to direct shear wave:
    u_theta_t_a_scan_m = get_u_theta_t_m(
        time_vector_s, radius_observation_m, phi_deg,
        s_l_sperm, s_t_sperm)

    # Compute polar displacement due to transverse head wave:
    # A bit weird to get the polar displacement from the head wave,
    # but it's what Bernstein and Spicer would have observed in their
    # 2000 paper using their semi-cylindrical sample.
    u_theta_tl_a_scan_m = get_u_theta_tl_m(
        time_vector_s, radius_observation_m, phi_deg,
        s_l_sperm, s_t_sperm)

    # The longitudinal wave should contribute nothing to u_theta.

    return (
        u_theta_t_a_scan_m,
        u_theta_tl_a_scan_m
    )


def get_u_v_displacement_components_at_iso_t_xz_plane_snapshot(
    X_grid_m,
    Z_grid_m,
    t_s,
    s_l_sperm,
    s_t_sperm
):
    """Get the u and v (x and y) components of displacement for
    a 2D grid of coordinate points in the XY plane (XY plane in B&S
    coordinate system, XZ plane in M. Riding coordinate system) at a
    specified snapshot in time.
    Assumes that the source is in the top-left corner of the coordinate grids
    (0,0).

    Parameters
    ----------
    X_grid_m : 2D array of x-coordinate values.  Units: Metres.
    Z_grid_m : 2D array of z-coordinate values.  Units: Metres.
    t_s : Snapshot time: time elapsed since source excitation at t=0.
        Units: Seconds.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    tuple
    (u_total_grid, v_total_grid)
    Where u and v are the x and z displacement components due to
    the direct shear, head shear and direct logitudinal waves combined.
    """
    # Transform from cartesian to circular polar coordinates:
    # Calculate radial distance from origin (0, 0):
    r_grid_m = np.sqrt(X_grid_m**2 + Z_grid_m**2)
    # Calculate angle from surface (phi) for all grid points:
    phi_grid_deg = np.rad2deg(np.atan2(Z_grid_m, X_grid_m))

    # Retrieve u & v (x, z) displacement components for all
    # wave modes:
    # Direct shear wave:
    u_t_grid_m = get_u_t_m(t_s, r_grid_m, phi_grid_deg, s_l_sperm, s_t_sperm)
    v_t_grid_m = get_v_t_m(t_s, r_grid_m, phi_grid_deg, s_l_sperm, s_t_sperm)
    # Shear head wave:
    u_tl_grid_m = get_u_tl_m(t_s, r_grid_m, phi_grid_deg, s_l_sperm, s_t_sperm)
    v_tl_grid_m = get_v_tl_m(t_s, r_grid_m, phi_grid_deg, s_l_sperm, s_t_sperm)
    # Longitudinal wave:
    u_l_grid_m = get_u_l_m(t_s, r_grid_m, phi_grid_deg, s_l_sperm, s_t_sperm)
    v_l_grid_m = get_v_l_m(t_s, r_grid_m, phi_grid_deg, s_l_sperm, s_t_sperm)

    # Sum x and z displacement components:
    u_total_grid_m = u_t_grid_m + u_tl_grid_m + u_l_grid_m
    v_total_grid_m = v_t_grid_m + v_tl_grid_m + v_l_grid_m

    return u_total_grid_m, v_total_grid_m


def get_time_vectors_around_direct_shear_arrival(
    radius_observation_m,
    s_t_sperm,
    dt_around_arrival_s,
    n_times,
):
    # The arrival time of the direct shear wave at the chosen radial
    # distance does not depend on the angle phi.
    t_arrival_direct_shear_s = radius_observation_m * s_t_sperm

    t_start_s = t_arrival_direct_shear_s - dt_around_arrival_s
    t_stop_s = t_arrival_direct_shear_s + dt_around_arrival_s

    time_vector_s = np.linspace(t_start_s, t_stop_s, n_times)

    # Repeat the same time vector once for every angle, forming an array:
    time_vectors_s = np.tile(time_vector_s, (n_angles, 1))

    return time_vectors_s


def get_time_vectors_around_head_wave_arrival(
    radius_observation_m,
    s_l_sperm,
    s_t_sperm,
    dt_around_arrival_s,
    n_times,
    theta_vector_deg,
):
    # Convert thetas to phis:
    phi_vector_deg = theta2phi(theta_vector_deg)

    # Pre-allocate a 2D numpy array in which to store the
    # time vectors for each angle:
    time_vectors_s = np.zeros((n_angles, n_times))
    for i, phi_i_deg in enumerate(phi_vector_deg):
        time_arrival_head_i_s = (
            (s_l_sperm *
             radius_observation_m *
             np.cos(np.deg2rad(phi_i_deg))
             ) +
            (radius_observation_m *
             np.sqrt((s_t_sperm**2) - (s_l_sperm**2)) *
             np.sin(np.deg2rad(phi_i_deg))
             )
        )
        t_start_s = time_arrival_head_i_s - dt_around_arrival_s
        t_stop_s = time_arrival_head_i_s + dt_around_arrival_s

        time_vectors_s[i] = np.linspace(t_start_s, t_stop_s, n_times)

    return time_vectors_s


def get_u_theta_a_scans_at_angles(
    theta_vector_deg,
    dt_around_arrival_s,
    wave_arrival_timed,
    n_times,
    radius_observation_m: float,
    s_l_sperm: float,
    s_t_sperm: float,
):
    # Get one vector of absolute time values since the excitation
    # for each angle in the theta_vector:
    time_vectors_s = get_time_vectors(radius_observation_m,
                                      s_l_sperm,
                                      s_t_sperm,
                                      dt_around_arrival_s,
                                      wave_arrival_timed,
                                      n_times,
                                      theta_vector_deg)

    # Create numpy arrays to store the output polar displacements
    # due to each wave type separately:
    u_theta_t_array_m = np.zeros((n_times, n_angles))
    u_theta_tl_array_m = np.zeros((n_times, n_angles))

    # For loop over angle theta, making one A-scan per iteration:
    for index_theta, theta_i_deg in enumerate(theta_vector_deg):
        # Get the time vecotr for this angle from the pre-computed array:
        time_vector_i_s = time_vectors_s[index_theta]
        # Get A-scans of displacement due to direct and
        # head shear waves separately for this angle:
        (u_theta_t_array_m[:, index_theta],
         u_theta_tl_array_m[:, index_theta]
         ) = get_u_theta_a_scans_at_angle(
             theta_i_deg, radius_observation_m, time_vector_i_s,
             s_l_sperm, s_t_sperm)

    return (
        u_theta_t_array_m,
        u_theta_tl_array_m)


def get_time_vectors(
    radius_observation_m: float,
    s_l_sperm: float,
    s_t_sperm: float,
    dt_around_arrival_s: float,
    wave_arrival_timed: str,
    n_times: int,
    theta_vector_deg,
):
    # Create array of time values to evaluate displacements at:
    if wave_arrival_timed == 'direct':
        time_vectors_s = get_time_vectors_around_direct_shear_arrival(
            radius_observation_m, s_t_sperm, dt_around_arrival_s, n_times
        )
    if wave_arrival_timed == 'head':
        time_vectors_s = get_time_vectors_around_head_wave_arrival(
            radius_observation_m, s_l_sperm, s_t_sperm, dt_around_arrival_s,
            n_times, theta_vector_deg
        )
    return time_vectors_s


def get_direct_and_head_u_theta_arrays_periodic_angles(
    radius_observation_m: float,
    c_l_mpers: float,
    c_t_mpers: float,
    dt_around_arrival_s: float,
    wave_arrival_timed: str,
    n_times: int,
    theta_start_deg: float,
    theta_stop_deg: float,
    n_angles: int,
):
    """
    Evaluates the displacement due to the direct and head shear waves at
    a constant observation radius from the source, over a vector of times
    from -dt to +dt around the arrival time of either the direct or head
    shear wave, and a vector of angles.

    Returns two arrays with time along the vertical axes (along a column)
    and angle theta along the horiontal axis (0-90 degrees) and the vectors
    of time and theta values used.

    REMEMBER: PHI IS ANGLE FROM SURFACE.  THIS IS CALLED THETA IN THE SYSTEM OF
    BERNSTEIN AND SPICER.
    HERE: IN THE SYSTEM OF RIDING, THETA IS THE ANGLE FROM THE SURFACE NORMAL.
    """
    # Calculate slownesses:
    s_l_sperm = 1 / c_l_mpers
    s_t_sperm = 1 / c_t_mpers

    # Create vector of angles to evaluate displacements at:
    theta_vector_deg = np.linspace(theta_start_deg,
                                   theta_stop_deg,
                                   n_angles)

    # Get the A-scan displacments:
    (u_theta_t_array_m,
     u_theta_tl_array_m) = get_u_theta_a_scans_at_angles(theta_vector_deg,
                                                         dt_around_arrival_s,
                                                         wave_arrival_timed,
                                                         n_times,
                                                         radius_observation_m,
                                                         s_l_sperm,
                                                         s_t_sperm)

    return (
        theta_vector_deg,
        u_theta_t_array_m,
        u_theta_tl_array_m)


def get_bernspice_ascan_contributions_in_polar_coordinates(
    time_vector_global_s,
    r_m,
    phi_deg,
    s_l_sperm,
    s_t_sperm
):
    """Returns A-scans of:
    1. the radial component of displacement due to the L wave.
    2. the theta component of displacement due to the direct shear wave.
    3. the theta crit component of displacement due to the head shear wave.
    over the time basis time_vector_global_s.
    Displacement components recorded at coordinate point (phi, r).

    Parameters
    ----------
    time_vector_global_s : Vector of time values to evaluate displacement
        components at.  Units: seconds
    r_m : Radial coordinate of evaluation point.  Units: metres.
    phi_deg : Angle from surface.  Units: degrees.
    s_l_sperm : Longitudinal wave slowness. Units: seconds per metre.
    s_t_sperm : Transverse wave slowness. Units: seconds per metre.

    Returns
    -------
    tuple
    (u_r_due_to_l_wave_ascan_m,
    u_theta_due_to_direct_s_wave_ascan_m,
    u_theta_crit_due_to_head_s_wave_ascan_m)
    """
    # Get (u, v) components of displacement in Bernstein & Spicer coodinates:

    # Longitudinal wave A-scans:
    u_l_ascan_m = get_u_l_m(time_vector_global_s,
                            r_m,
                            phi_deg,
                            s_l_sperm,
                            s_t_sperm)
    v_l_ascan_m = get_v_l_m(time_vector_global_s,
                            r_m,
                            phi_deg,
                            s_l_sperm,
                            s_t_sperm)

    # Direct shear wave A-scans:
    u_s_ascan_m = get_u_t_m(time_vector_global_s,
                            r_m,
                            phi_deg,
                            s_l_sperm,
                            s_t_sperm)
    v_s_ascan_m = get_v_t_m(time_vector_global_s,
                            r_m,
                            phi_deg,
                            s_l_sperm,
                            s_t_sperm)

    # Head wave A-scans:
    u_head_ascan_m = get_u_tl_m(time_vector_global_s,
                                r_m,
                                phi_deg,
                                s_l_sperm,
                                s_t_sperm)
    v_head_ascan_m = get_v_tl_m(time_vector_global_s,
                                r_m,
                                phi_deg,
                                s_l_sperm,
                                s_t_sperm)

    # Resolve the radial component of displacement by l-wave
    # from Bernstein & spicer (u, v) frame:
    (u_r_due_to_l_wave_ascan_m
     ) = resolve_u_r_from_bernstein_u_and_v(u_l_ascan_m,
                                            v_l_ascan_m,
                                            phi_deg)

    # Resolve theta component of displacement by direct s-wave
    # from Bernstein & spicer (u, v) frame:
    (u_theta_due_to_direct_s_wave_ascan_m
     ) = resolve_u_theta_from_bernstein_u_and_v(u_s_ascan_m,
                                                v_s_ascan_m,
                                                phi_deg)

    # Resolve theta* component of displacement by head s-wave
    # from Bernstein & spicer (u, v) frame:
    (u_theta_crit_due_to_head_s_wave_ascan_m
     ) = resolve_u_theta_crit_from_bernstein_u_and_v(
         u_head_ascan_m,
         v_head_ascan_m,
         1 / s_l_sperm,
         1 / s_t_sperm)

    return (
        u_r_due_to_l_wave_ascan_m,
        u_theta_due_to_direct_s_wave_ascan_m,
        u_theta_crit_due_to_head_s_wave_ascan_m
    )


# %%
if __name__ == '__main__':
    # Generate data:

    # Input constants:
    # s_l: Slowness of the longtudinal wave.  Reciprocal of longitudinal
    # wave speed c_l.
    c_l_mpers = 6370
    s_l_sperm = 1 / c_l_mpers

    # s_t: Slowness of the transverse wave.  Reciprocal of transverse
    # wave speed c_t.
    c_t_mpers = 3130
    s_t_sperm = 1 / c_t_mpers

    # Evaluate displacement A-scans for a vector of eually-spaced angles:

    # at r_m:
    radius_observation_m = 40 * 10**-3
    dt_s = 5 * 10**-6
    n_times = 500
    (time_vector_relative_to_direct_arrival_s
     ) = np.linspace(-dt_s, dt_s, n_times)

    # Angle will go from theta=0 (phi=90) to theta=90 (phi=0) degrees:
    theta_start_deg = 0
    theta_stop_deg = 90
    n_angles = 100

    # Evaluate displacements:
    (theta_vector_deg,
     u_theta_t_array_m,
     u_theta_tl_array_m
     ) = get_direct_and_head_u_theta_arrays_periodic_angles(
         radius_observation_m,
         c_l_mpers,
         c_t_mpers,
         dt_s,
         'direct',
         n_times,
         theta_start_deg,
         theta_stop_deg,
         n_angles)

    # %% Visualisation:
    # Plot a colormap of the displacements as a function of angle and time:
    fig_width_in = 5
    fig_height_in = 12
    c_map = cc.m_CET_D7
    max_abs_u_cmap_m = 2 * 10**-9

    # ax_1 (top): colormap of direct shear displacements
    # ax_2: colormap of transverse head wave displacements
    # ax_3: colormap of direct + head wave displacements
    # ax_4: direct shear radiation pattern
    (fig,
     (ax_u_direct,
      ax_u_head,
      ax_u_total,
      ax_radpat)) = plt.subplots(4, 1, layout='constrained',
                                 figsize=(fig_width_in, fig_height_in))
    # Direct shear displacements:
    im_u_theta_t = ax_u_direct.imshow(u_theta_t_array_m, cmap=c_map,
                                      extent=(theta_start_deg, theta_stop_deg,
                                              dt_s, -dt_s),
                                      aspect='auto',
                                      vmin=-max_abs_u_cmap_m,
                                      vmax=max_abs_u_cmap_m)
    ax_u_direct.set_xlabel(r'Emission angle $\theta$ (°)')
    ax_u_direct.set_ylabel('Time relative to arrival time (s)')
    fig.colorbar(im_u_theta_t, ax=ax_u_direct,
                 label=r'Displacement $u_{\theta\ T}$ (Arb.)')
    ax_u_direct.set_title('Polar direct shear wave displacements:')

    # Head wave displacements:
    im_u_theta_tl = ax_u_head.imshow(u_theta_tl_array_m, cmap=c_map,
                                     extent=(theta_start_deg, theta_stop_deg,
                                             dt_s, -dt_s),
                                     aspect='auto',
                                     vmin=-max_abs_u_cmap_m,
                                     vmax=max_abs_u_cmap_m)
    ax_u_head.set_xlabel(r'Emission angle $\theta$ (°)')
    ax_u_head.set_ylabel('Time relative to arrival time (s)')
    fig.colorbar(im_u_theta_tl, ax=ax_u_head,
                 label=r'Displacement $u_{\theta\ TL}$ (Arb.)')
    ax_u_head.set_title('Polar head wave displacements:')

    # Combined waveform: direct + head wave displacements:
    im_u_theta_combined = ax_u_total.imshow((u_theta_t_array_m +
                                             u_theta_tl_array_m),
                                            cmap=c_map,
                                            extent=(theta_start_deg,
                                                    theta_stop_deg,
                                                    dt_s, -dt_s),
                                            aspect='auto',
                                            vmin=-max_abs_u_cmap_m,
                                            vmax=max_abs_u_cmap_m)
    ax_u_total.set_xlabel(r'Emission angle $\theta$ (°)')
    ax_u_total.set_ylabel('Time relative to arrival time (s)')
    fig.colorbar(im_u_theta_combined, ax=ax_u_total,
                 label=r'Displacement $u_{\theta}$ (Arb.)')
    ax_u_total.set_title('Polar displacements due to direct shear and' +
                         'head wave combined:')

    # # Plot the heaviside window for the head wave:
    # im_heaviside = ax_heaviside.imshow(h_tl_minus_h_st_array,
    #                                    cmap=c_map,
    #                                    extent=(theta_start_deg,
    #                                            theta_stop_deg,
    #                                            dt_after_s, -dt_s),
    #                                    aspect='auto')
    # ax_heaviside.set_xlabel(r'Emission angle $\theta$ (°)')
    # ax_heaviside.set_ylabel('Time relative to arrival time (s)')
    # fig.colorbar(im_heaviside, ax=ax_heaviside,
    #              label=r'Heaviside output')
    # ax_heaviside.set_title('Heaviside window:')

    # Plot the direct shear wave radiation pattern on the bottom axes:
    row_index = int((dt_s / (2*dt_s)) * n_times) + 1
    rad_pattern = u_theta_t_array_m[row_index]
    ax_radpat.plot(theta_vector_deg, rad_pattern)
    ax_radpat.set_xlim([theta_start_deg,
                        theta_stop_deg])
    ax_radpat.axhline(0, color='k', lw=0.6)

    # Highlight the row of the radiation pattern:
    time_rad_pat_s = time_vector_relative_to_direct_arrival_s[row_index]

    ax_u_direct.axhline(time_rad_pat_s, 0, 0.05, ls='--', color='lime')

    # %% A-scans:

    # # I should be able to reproduce the integral of the du_theta/dt A-scans
    # # presented by Bernstein and Spicer (See figure 7 of Bernstein and Spicer
    # # (2000)).  Note that, in Bernstein's thesis, she explains how the
    # # displacement over time A-scans were computed, and then differentiated
    # # to obtain the signal that their EMAT detector would output.
    # list_of_thetas_A_scans_deg = [18, 35, 45, 60, 88]

    # # Create a new figure:
    # (fig_2, ax_ascans
    # ) = plt.subplots(1, 1, layout='constrained', figsize=(6, 5))
    # ax_ascans.set_xlabel('Time relative to direct T arrival (μs)')
    # ax_ascans.set_ylabel('Displacement (Arb.)')
    # ax_ascans.set_yticks([])
    # # Make y-axs span from 0 to 1:
    # ax_ascans.set_ylim(0, 1)
    # # Set a scale factor that scales the displacements so they are visible:
    # u_scale_factor = 4 * 10**7
    # # Find the heights of each A-scan zero line:
    # n_a_scans = len(list_of_thetas_A_scans_deg)
    # y_step_m = 1 / (n_a_scans + 1)
    # y_zeros_m = np.linspace(y_step_m, (1 - y_step_m), n_a_scans)

    # for i, theta_a_scan_deg in enumerate(list_of_thetas_A_scans_deg):
    #     # Compute the A-scan for this angle:
    #     phi_i_deg = theta2phi(theta_a_scan_deg)
    #     u_theta_a_scan_m = np.zeros(n_times)
    #     for index_time, time_s in enumerate(time_vector_s):
    #         # Compute displacement due to direct shear wave:
    #         u_theta_t_a_scan_m = get_u_theta_t_m(time_s,
    #                                              radius_observation_m,
    #                                              phi_i_deg)
    #         # Compute displacement due to head shear wave:
    #         u_theta_tl_a_scan_m = get_u_theta_tl_m(time_s,
    #                                                radius_observation_m,
    #                                                phi_i_deg)
    #         # Sum the two contributions and store the result:
    #         u_theta_a_scan_m[index_time] = (u_theta_t_a_scan_m +
    #                                         u_theta_tl_a_scan_m)
    #     # Plot this A-scan:
    #     y_zero_m = y_zeros_m[i]
    #     u_theta_a_scan_transformed = ((u_theta_a_scan_m * u_scale_factor) +
    #                                    y_zero_m)
    #     ax_ascans.plot(time_vector_relative_to_direct_arrival_s / 10**-6,
    #                    u_theta_a_scan_transformed)
    #     ax_ascans.text(0.05, y_zero_m, f'{theta_a_scan_deg} deg',
    #                    va='bottom', transform=ax_ascans.transAxes)
    #     ax_ascans.axhline(y_zero_m, color='k', lw=0.6)

    #     # Highlight this A-scan on ax_1:
    #     ax_u_direct.axvline(theta_a_scan_deg, ls='--', color='w')

    plt.show()

    # %%
