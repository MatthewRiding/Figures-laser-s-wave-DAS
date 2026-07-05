import numpy as np


def calculate_b_mm_from_t_0_us(t_0_us, c_mpers):
    b_mm = (0.5 * (t_0_us * 10**-6) * c_mpers) / 10**-3
    return b_mm


def calculate_c_vert_HS(c_t_mpers, c_lsaw_mpers):
    c_vert_mpers = ((c_t_mpers * c_lsaw_mpers) /
                    np.sqrt(c_lsaw_mpers**2 - c_t_mpers**2))
    return c_vert_mpers


def calculate_b_mm_from_t_0_HS_us(t_0_HS_us, c_t_mpers, c_lsaw_mpers):
    # In the case of the H-S reflection in an NMO B-scan,
    # t_0 may be defined as the t-axis intercept of the straight line
    # fromed by the H-S reflection in the x-t domain.
    # The equation for the line can be expressed as:
    # t_HS(x) = (x / c_LSW) + t_0HS
    # Where x = offset, c_LSW = speed of leaky surface wave,
    # and t_0HS:
    # t_0_HS = 2b / c_vert
    # Where:
    # c_vert = (c_T * C_LSW) / sqrt(c_LSW^2 - c_T^2)
    c_vert_mpers = calculate_c_vert_HS(c_t_mpers, c_lsaw_mpers)
    b_mm = (0.5 * (t_0_HS_us * 10**-6) * c_vert_mpers) / 10**-3
    return b_mm


def calculate_t_0_HS_us_from_b_mm(b_mm, c_t_mpers, c_lsaw_mpers):
    c_vert_mpers = calculate_c_vert_HS(c_t_mpers, c_lsaw_mpers)
    t_0_HS_us = ((2 * (b_mm * 10**-3)) / c_vert_mpers) / 10**-6
    return t_0_HS_us
