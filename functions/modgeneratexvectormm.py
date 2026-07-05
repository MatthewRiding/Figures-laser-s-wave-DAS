import numpy as np


def generate_x_vector_mm(n_elements, pitch_mm):
    aperture_mm = (n_elements - 1) * pitch_mm
    x_vector_mm = np.linspace(0, aperture_mm, n_elements)
    return x_vector_mm
