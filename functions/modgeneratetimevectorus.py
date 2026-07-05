import numpy as np


def generate_time_vector_us(t_min_us, t_max_us, n_samples):
    time_vector_us = np.linspace(t_min_us, t_max_us, n_samples, endpoint=True)
    return time_vector_us
