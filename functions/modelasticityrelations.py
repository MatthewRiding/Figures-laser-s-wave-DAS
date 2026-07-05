import numpy as np


def kappa_from_v_poisson(v_poisson):
    # Relationship between kappa (c_L/c_T) and Poisson's ratio from page 124
    # of Achenbach: Wave propagation in elastic solids.
    kappa = np.sqrt((2 * (1-v_poisson)) / (1 - (2 * v_poisson)))
    return kappa
